"""Generate the Japanese experiment record from saved, validated measurements."""
import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics

from .analyze_factorial import interval, write_csv
from .benchmark_topology import control, dump, sha
from .topology_search import Candidate, Machine


def csv_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
                      *['|'+'|'.join(map(str,row))+'|' for row in rows]])


def label(c,machine):
    p = len(set(c.layer)&set(machine.p_primary))
    s = len(set(c.layer)&set(machine.p_smt))
    e = len(set(c.layer)&set(machine.e))
    split = '/'.join(str(len(set(c.layer)&set(g))) for g in machine.efficient_l2)
    output = f'{len(c.output)}人'
    return (f'{c.backend} P{p+s}E{e}(E={split}) {len(c.layer)}→{output}, '
            f'{"D" if c.dynamic else "S"}/{"D" if c.output_dynamic else "S"}, {4 if c.row4 else 1}行')


def generate(sweep,confirmation,online_raw,online_normalized,detail=None):
    machine_obj=json.loads((sweep/'machine.json').read_text())
    machine=Machine(tuple(map(tuple,machine_obj['performance'])),tuple(map(tuple,machine_obj['efficient_l2'])),tuple(machine_obj['low_power']))
    plan=json.loads((sweep/'plan.json').read_text())
    cs={c.name:c for c in map(Candidate.from_dict,plan['candidates'])}
    confirm_cells=csv_rows(confirmation/'cells.csv')
    finalists=[r for r in confirm_cells if r['kind']=='confirmation']
    finalists.sort(key=lambda r:float(r['mean_ms_per_token']))
    best=finalists[0]
    lookup={(r['name'],int(r['replicate'])):r for r in csv_rows(confirmation/'normalized_trials.csv')}
    reps=sorted({rep for _,rep in lookup})
    contrasts=[]
    for r in confirm_cells:
        logs=[math.log(float(lookup[r['name'],rep]['ms_per_token'])/float(lookup[best['name'],rep]['ms_per_token'])) for rep in reps]
        avg,lo,hi=interval(logs)
        contrasts.append({'name':r['name'],'reference':best['name'],'blocks':len(reps),
                          'best_speedup_pct':100*math.expm1(avg),
                          'ci95_low_pct':100*math.expm1(lo),'ci95_high_pct':100*math.expm1(hi)})
    write_csv(confirmation/'paired_vs_best.csv',contrasts)
    contrast_by_name={r['name']:r for r in contrasts}
    fusion=csv_rows(sweep/'fusion_contrasts.csv')
    fusion_group=[]
    for layer in sorted({r['layer'] for r in fusion}):
        for backend in ('openmp','pthread'):
            values={}
            for treatment in ('qkv_only','ffn_only','combined','synergy'):
                chosen=[r for r in fusion if r['layer']==layer and r['backend']==backend and r['treatment']==treatment]
                per_block=[statistics.mean(json.loads(r['block_saved_ms'])[i] for r in chosen) for i in range(3)]
                avg,lo,hi=interval(per_block)
                values[treatment]={'saved_ms':avg,'ci95_low':lo,'ci95_high':hi}
            fusion_group.append({'layer':layer,'backend':backend,'treatments':values})
    online=[]
    for source in (online_raw,online_normalized):
        obj=json.loads((source/'plan.json').read_text())
        selected=Candidate.from_dict(json.loads((source/'selection.json').read_text())['candidates'][0])
        runs=csv_rows(source/'trials.csv')
        candidate_runs=[r for r in runs if r['kind']=='online']
        online.append({'objective':obj['args']['objective'],'candidate':asdict(selected),'name':selected.name,
                       'measured_candidates':len(candidate_runs),'measurements':len(runs),
                       'candidate_wall_s':sum(float(r['wall_s']) for r in candidate_runs),
                       'total_run_wall_s':sum(float(r['wall_s']) for r in runs),
                       'confirmation_mean_tokens_per_second':next(float(r['mean_tokens_per_second']) for r in finalists if r['name']==selected.name),
                       'confirmation_latency_gap_pct':100*(float(next(r['mean_ms_per_token'] for r in finalists if r['name']==selected.name))/float(best['mean_ms_per_token'])-1)})
    raw_search=csv_rows(sweep/'raw_search_summary.csv')
    normalized_search=csv_rows(sweep/'search_summary.csv')
    counts=[]
    for source in (sweep,online_raw,online_normalized,confirmation,*([detail] if detail else [])):
        runs=csv_rows(source/'trials.csv')
        counts.append({'directory':source.name,'measurements':len(runs),
                       'candidates':sum(r['kind'] not in ('warmup','anchor') for r in runs),
                       'controls':sum(r['kind']=='anchor' for r in runs),
                       'warmups':sum(r['kind']=='warmup' for r in runs),
                       'sum_run_wall_s':sum(float(r['wall_s']) for r in runs)})
    stages=[]
    if detail:
        details=csv_rows(detail/'normalized_trials.csv')
        for name in sorted({r['name'] for r in details}):
            summaries=[json.loads((detail/r['path']/'summary.json').read_text()) for r in details if r['name']==name]
            def avg(keys):
                return statistics.mean(sum(s.get(k+'_ms_per_token',0.) for k in keys) for s in summaries)
            qkv=avg(('qkv',)) if 'qkv_ms_per_token' in summaries[0] else avg(('q_proj','k_proj','v_proj'))
            ffn=avg(('ffn_up_gate_swiglu','ffn_down')) if 'ffn_up_gate_swiglu_ms_per_token' in summaries[0] else avg(('ffn_up','ffn_gate','swiglu','ffn_down'))
            stages.append({'name':name,'runs':len(summaries),'qkv_ms_per_token':qkv,
                           'ffn_including_swiglu_down_ms_per_token':ffn,'output_ms_per_token':avg(('output',)),
                           'layer_total_ms_per_token':avg(('layer_total',)),'forward_ms_per_token':avg(('forward',)),
                           'forward_calls':summaries[0]['forward_calls'],'measured_tokens':summaries[0]['measured_tokens']})
        write_csv(detail/'stage_summary.csv',stages)
    metrics={'best_confirmation':best,'best_candidate':asdict(cs[best['name']]),'paired':contrasts,'stages':stages,
             'fusion':fusion_group,'online':online,'raw_search':raw_search,'normalized_search':normalized_search,'costs':counts}
    dump(sweep/'report_metrics.json',metrics)
    anchor=control(machine)
    profile_specs=[replace(anchor,qkv=q,ffn=f) for q in (0,1) for f in (0,1)]
    profile_specs.extend([cs[best['name']],Candidate.from_dict(online[0]['candidate']),
                          cs[next(r['name'] for r in finalists if cs[r['name']].backend=='openmp')]])
    dump(sweep/'selection_profile.json',{'rule':'four QKV/FFN control cells + best confirmation + raw online winner + best OpenMP finalist',
                                        'candidates':[asdict(c) for c in dict.fromkeys(profile_specs)]})
    raw48=next(r for r in raw_search if r['method']=='model' and r['budget']=='48')
    random48=next(r for r in raw_search if r['method']=='random' and r['budget']=='48')
    blocks=csv_rows(sweep/'blocks.csv')
    texts=[f'# 並列化単体・複合とトポロジー構成探索（2026-10-22）',
           f'480採用候補を実測し、QKV/FFN単独と併用の追加48セルも比較した。'
           f'独立8ブロックの確認で最速平均だった候補は **{label(cs[best["name"]],machine)}**、'
           f'**{float(best["mean_tokens_per_second"]):.2f} tok/s**。'
           f'この機械・stories15M・256steps・今回の有限空間に対する結果である。',
           f'トポロジー予測探索は48候補で、別ブロック最良値への平均latency差が'
           f' **{float(raw48["mean_heldout_regret_pct"]):.2f}%**、ランダム探索は'
           f' **{float(random48["mean_heldout_regret_pct"]):.2f}%**。'
           f'ただし3ブロックを再利用するreplayであり、60件の独立実測ではない。',
           '## 計画と実施範囲',
           '機械はCore Ultra 5 225U、P物理2個（SMT計4）、E8個（L2ごと4個）、LP-E2個。'
           'PのSMT有無、E0/2/4/6/8個、compact/spread、outputだけCPU追加を含む30配置に、'
           '管理方式・層分配・output分配・1/4行を交差した480候補。'
           '融合は採用探索では有効にし、2配置でQKV/FFNの4組合せを別に検証した。',
           '採用空間と事前固定した探索法は [TOPOLOGY_SEARCH.md](../../src/cpuprofiler/TOPOLOGY_SEARCH.md)。'
           '`dev/wataru` のL2占有分割と `dev/cpu-profiler` のトポロジー取得を活かし、'
           'CPUの追加だけでなくSMTを外す入替・output人数変更も扱う。'
           '任意のCPU部分集合、重み、任意chunk、バリア実装は今回の空間外。',
           '全実験でstories15M、temperature=0、seed42、同じprompt、steps256。'
           '256は上限で、詳細計測では区切りtokenで222 forward呼出しで終了し、先頭を除く221回を集計した。'
           '速度測定はprofileなし。各完全ブロック内で順番をシャッフルし、'
           '温度・周波数snapshot・負荷・電源状態と実行前後の時刻を保存した。'
           '未改造upstreamとqkv-ffnブランチを同じコンパイル条件で再ビルドして含めた。',
           table(['実験','候補/比較測定','control','warmup','合計'],
                 [[r['directory'],r['candidates'],r['controls'],r['warmups'],r['measurements']] for r in counts]),
           '## QKV・FFN単独と併用',
           '以下は各配置・方式でstatic/dynamicと1/4行を均等平均した、両方なしからの短縮ms/token。'
           '周辺効果とは違い、QKV単独とFFN単独は両方なしを基準にする。'
           'FFN処置はup/gate融合＋SwiGLU並列化を含む。両者を独立には分離していない。',
           table(['配置','方式','QKV単独','FFN単独','併用','相乗短縮'],
                 [[('P4E4 spread' if len(r['layer'].split(','))==8 else 'P2E8'),r['backend'],
                   *[f"{r['treatments'][k]['saved_ms']:+.4f}" for k in ('qkv_only','ffn_only','combined','synergy')]] for r in fusion_group]),
           '相乗短縮=T(Qのみ)+T(Fのみ)−T(併用)−T(なし)。正なら単独短縮の合計より速い。'
           '今回も併用の平均改善は確認できるが、相乗の大きさは条件依存である。'
           '[全64対比較](fusion_contrasts.csv)には条件別の値、3ブロックの差、探索的bootstrap区間を保存した。'
           '3ブロックの区間を確証的な有意差として扱わない。',
           '## 独立した上位確認',
           '探索の補正順位上位4、生値順位上位4、方式/SMT別代表、従来control、実際のオンライン探索の推薦を'
           '測定前に固定し、別の8ブロックで比較した。下表は生の平均速度。'
           'S=static、D=dynamic（層16物理行、output32物理行）、Pの数字はSMT込み論理CPU数。',
           table(['候補','tok/s','ms/token'],
                 [[label(cs[r['name']],machine) if r['name'] in cs else r['name'],
                   f"{float(r['mean_tokens_per_second']):.2f}",f"{float(r['mean_ms_per_token']):.4f}"]
                  for r in sorted(confirm_cells,key=lambda r:float(r['mean_ms_per_token']))]),
           f'最速平均候補の実行設定は [採用候補JSON](best_config.json)。'
           '「真の最適」の保証はなく、この確認自体でも複数候補から最良を選ぶバイアスが残る。']
    for name in ('upstream','qkvffn_branch'):
        r=contrast_by_name[name]
        texts.append(f'{name}に対する速度改善はブロック内のlatency比の幾何平均で'
                     f' **{r["best_speedup_pct"]:.2f}%**、探索的95%区間'
                     f' [{r["ci95_low_pct"]:.2f}, {r["ci95_high_pct"]:.2f}]%。'
                     '各比較対象の未改造ソースを同じコンパイル条件でビルドした実装全体の比較であり、'
                     '差を単一の要因に帰属しない。'
                     + ('区間がゼロをまたぐため、この比較の改善は確定しない。' if r['ci95_low_pct']<=0<=r['ci95_high_pct'] else ''))
    texts.extend([
        '## 探索アルゴリズムの比較',
        '同じ480候補を対象に、random、最良点からの1因子近傍探索（beam幅1、5回に1回再始動）、'
        'カーネルのみのridge予測器、トポロジーを含むridge予測器を比較した。'
        '予測器は最初の12点を特徴空間で分散配置し、指定した相互作用・E数の二次項も使う。'
        'その後は予測平均−0.05×探索指標の点を選び、5回に1回は一様探索する。'
        '推薦は実際に照会した構成の最良観測値に限定する。',
        '3ブロックのleave-one-block-outで、2ブロック平均を探索への観測、残り1ブロックを評価に使った。'
        'seed0〜19、各12/24/48/96照会。以下は生のms/tokenでの結果。'
        'regretは「推薦構成の評価ブロックlatency／その評価ブロック全480候補の最小latency−1」。'
        '単一ブロックの最小値も測定雑音で楽観的になり得る。',
        table(['探索法','候補照会','平均差%','p90差%','3%以内到達率'],
              [[r['method'],r['budget'],f"{float(r['mean_heldout_regret_pct']):.2f}",
                f"{float(r['p90_heldout_regret_pct']):.2f}",f"{100*float(r['within_3pct_rate']):.1f}%"]
               for r in raw_search]),
        f'48/480候補（候補照会90%削減）でトポロジー予測器は3%以内へ'
        f' **{100*float(raw48["within_3pct_rate"]):.1f}%** 到達した。'
        '予測器からトポロジーを除くと性能が落ち、このデータでは構成の特徴を使う意義が見える。'
        'ただし12点ではトポロジーなし予測器より良くなく、初期設計の費用は小さくない。'
        '480候補の総当たりも訓練ブロックで選んで評価し、評価ブロック最良を直接推薦していない。',
        f'48照会時の予測器処理は平均{float(raw48["mean_search_cpu_s"]):.3f} CPU秒。'
        f'候補実行時間は訓練2ブロックの合計で平均{float(raw48["mean_simulated_run_wall_s"]):.2f}秒だった。'
        'これらはcontrol・warmup・ビルド・独立確認を含まない。',
        '[全照会trace](raw_search_trace.csv)、[各fold/seedの結果](raw_search_runs.csv)、'
        '[集計](raw_search_summary.csv)、[評価プロトコル](raw_replay_protocol.json)。'
        '60件は同じ3ブロックから作った探索順の反復であり、60回の独立ベンチマークとは解釈しない。',
        '## 粗い間隔の補正の失敗と実測オンライン探索',
        '当初は16セルごとのcontrolを時刻で線形補間して割る方針だった。'
        f'しかし第1ブロックのcontrolは {float(blocks[0]["min_control_ms"]):.3f}〜'
        f'{float(blocks[0]["max_control_ms"]):.3f}ms/token と揺れ、一時的なcontrolの遅れが'
        '周辺候補を過度に有利にした。温度・負荷のsnapshotは残したが、原因は断定しない。',
        table(['探索法','照会','生値regret%','補正後regret%'],
              [[r['method'],r['budget'],f"{float(r['mean_heldout_regret_pct']):.2f}",
                f"{float(next(n['mean_heldout_regret_pct'] for n in normalized_search if n['method']==r['method'] and n['budget']==r['budget'])):.2f}"]
               for r in raw_search if r['budget'] in ('48','480')]),
        '補正後のheld-out最良値自体も変わるため、上の2列のregretは同じ物差しではない。'
        'この16セル間隔の単純補正を主な採用基準にはしない。生値も評価する方針への変更は感度分析を見てからの判断であり、'
        '事前固定とは区別する。補正値・失敗結果・最初の計画も消さずに保存した。',
        table(['実測探索の目的','48候補の推薦','確認時tok/s','確認最速とのlatency差','候補実行秒','control等込み秒'],
              [[r['objective'],label(Candidate.from_dict(r['candidate']),machine),
                f"{r['confirmation_mean_tokens_per_second']:.2f}",f"{r['confirmation_latency_gap_pct']:.2f}%",
                f"{r['candidate_wall_s']:.2f}",f"{r['total_run_wall_s']:.2f}"] for r in online]),
        '各オンライン実測は候補48回＋前後control49回＋warmup2回＝99回。'
        '全480候補の既存速度表は探索器へ渡していない。上の秒数はsubprocess実行時間の合計で、'
        'コンパイル、Python処理、ファイル保存、独立確認の費用を含まない。'
        'replayでは1観測に訓練2ブロックを平均するため48照会=候補96測定で、'
        'オンラインの1候補1回と同じ測定条件ではない。',
        '今回のオンライン探索では、粗い間隔の補正と違って候補ごとの前後にcontrolを置いた補正版が、'
        '独立確認で最速平均の構成を見つけた。replayとオンラインは対照の間隔・測定回数・時期が異なるため、'
        '補正全般が悪いとも、生値目的が常に良いとも結論しない。各オンライン目的はseed0の1回だけである。'
        'トポロジー予測探索48点→生値と局所対照比の両方で上位候補を残す→独立確認、を今回の推奨とする。'
        '一点の比だけで採用を確定しない。'
        '次に試すならcontrolの複数反復/robust平滑化と、最良点の再測定を予算内に組み込む。'
        'これらの改良案は今回まだ検証していない。',
        '## 再現・検証と限界',
        '標準ライブラリのみで測定・探索・集計を実行できる。'
        '生データの欠損/重複、全stdout hash、summaryとCSVの値、ソースhash、'
        'controlの前後挟み込みを検証した。Cの数値一致と探索の因果性・ridge更新・'
        '補正/相互作用の算術を含むテストの結果は [tests.log](tests.log)。',
        '全実験を再検査した結果は [final_audit.json](final_audit.json)。'
        '`python3 results/topology_adoption_20261022/verify.py` で再検査できる。',
        'tok/sはC側の既存タイマーで先頭反復を除き、forwardに加えてsampling/decode/出力を含む。'
        'モデル読込・ビルド・プロセス起動を含まない。タイマーはミリ秒単位である。'
        '個々の構成の短い測定に雑音があり、温度・周波数・'
        '直前の構成の影響を完全には分離できない。今回の探索空間で有望でも、大型モデル、'
        '長い文脈、別CPU、複数リクエストでの最適性は未確認。'
        'bootstrap区間は少ないブロックでの探索的な値で、多重比較を補正していない。',
        '```sh\n'
        'PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --output results/NEW_sweep --repeats 3\n'
        'PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_sweep --raw-replay\n'
        'PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --online --objective raw --budget 48 --output results/NEW_online_raw\n'
        'PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --online --objective normalized --budget 48 --output results/NEW_online_normalized\n'
        'PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_sweep --no-replay --include-online results/NEW_online_raw --include-online results/NEW_online_normalized\n'
        'PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --confirm results/NEW_sweep --output results/NEW_confirm --repeats 8\n'
        'PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_confirm --no-replay\n'
        'PYTHONPATH=src python3 -m cpuprofiler.report_topology results/NEW_sweep results/NEW_confirm results/NEW_online_raw results/NEW_online_normalized\n'
        'PYTHONPATH=src python3 -m cpuprofiler.benchmark_topology --confirm results/NEW_sweep --output results/NEW_detail --repeats 3 --profile\n'
        'PYTHONPATH=src python3 -m cpuprofiler.analyze_topology results/NEW_detail --no-replay\n'
        'PYTHONPATH=src python3 -m cpuprofiler.report_topology results/NEW_sweep results/NEW_confirm results/NEW_online_raw results/NEW_online_normalized --detail results/NEW_detail\n'
        '```',
        '結果フォルダにはplan、machine、environment、実行設定、状態snapshot、stdout/stderr、'
        '生CSV、ソースsnapshot、ビルドコマンド、モデル/tokenizer/バイナリhashを残した。'
        '`analysis_source/` と `analysis_provenance.json` は集計時のPythonソースと入力hash。'
        '`report_metrics.json` とこの報告は `report_topology.py` から再生成できる。'
        '比較ブランチの解決済みcommitは `builds/*/provenance.json` に記録。'])
    if detail:
        texts.insert(-3,'## 演算別の追加計測\n\n'
                     f'詳細計測は [{detail.name}](../{detail.name}/stage_summary.csv) に別保存した。'
                     '同じCPU配置とカーネルでQKV/FFNの4組合せを測り、上位構成も比較した。'
                     '値はms/token、各3回平均。FFNはSwiGLUとdownを含む。層別CSVと圧縮JSONLも保存した。'
                     'profileありの時計取得・記録・バリア費用を含むため、速度ランキングに混ぜない。'
                     'layer_totalと各stageは重複するため合計しない。\n\n'
                     +table(['構成','QKV','FFN','output'],
                            [[label(cs[r['name']],machine)+f" Q{cs[r['name']].qkv}F{cs[r['name']].ffn}",
                              f"{r['qkv_ms_per_token']:.4f}",f"{r['ffn_including_swiglu_down_ms_per_token']:.4f}",
                              f"{r['output_ms_per_token']:.4f}"] for r in stages]))
    (sweep/'README.md').write_text('\n\n'.join(texts)+'\n')
    dump(sweep/'best_config.json',{'candidate':asdict(cs[best['name']]),
         'confirmation_mean_tokens_per_second':float(best['mean_tokens_per_second']),
         'backend':cs[best['name']].backend,'cpus':cs[best['name']].cpus,
         'threads':len(cs[best['name']].layer),'output_threads':len(cs[best['name']].output),
         'omp_schedule':'dynamic,16' if cs[best['name']].dynamic else 'static',
         'output_dynamic':bool(cs[best['name']].output_dynamic),'output_chunk':32,
         'output_block':4 if cs[best['name']].row4 else 1,'p_weight':1,'fuse_qkv':True,'fuse_ffn':True})
    dump(sweep/'report_provenance.json',{'script_sha256':sha(Path(__file__)),
         'inputs':{str(p):sha(p) for p in (sweep/'fusion_contrasts.csv',sweep/'raw_search_summary.csv',
                   sweep/'search_summary.csv',confirmation/'cells.csv',confirmation/'normalized_trials.csv',
                   online_raw/'selection.json',online_normalized/'selection.json')}})
    print(f'Report: {sweep/"README.md"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sweep','confirmation','online_raw','online_normalized'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--detail',type=Path)
    args=parser.parse_args()
    generate(args.sweep,args.confirmation,args.online_raw,args.online_normalized,args.detail)


if __name__ == '__main__': main()
