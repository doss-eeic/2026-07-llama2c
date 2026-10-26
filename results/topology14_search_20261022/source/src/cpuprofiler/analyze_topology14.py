"""Validate and report full-depth topology experiments without hiding the scope."""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

from .analyze_factorial import interval,write_csv
from .benchmark_topology import control,dump,sha
from .benchmark_topology14 import METHODS,mirror,prior_best
from .topology14 import Space,Policy
from .topology_search import Candidate,Machine


def read_csv(path):
    with path.open() as file: return list(csv.DictReader(file))


def validate(root):
    plan=json.loads((root/'plan.json').read_text())
    machine=json.loads((root/'machine.json').read_text())
    machine=Machine(tuple(map(tuple,machine['performance'])),tuple(map(tuple,machine['efficient_l2'])),tuple(machine['low_power']))
    space=Space(machine)
    rows=read_csv(root/'trials.csv')
    if not (root/'complete.json').exists() or len(rows)!=json.loads((root/'complete.json').read_text())['measurements']:
        raise ValueError(f'Incomplete run: {root}')
    specs={Candidate.from_dict(c).name:Candidate.from_dict(c) for c in plan['candidates']}
    if len(specs)!=len(plan['candidates']): raise ValueError('Duplicate plan candidates')
    if [int(r['sequence']) for r in rows]!=list(range(1,len(rows)+1)): raise ValueError('Duplicate/missing run sequence')
    digests=set()
    for r in rows:
        for k in ('sequence','replicate','order'): r[k]=int(r[k])
        for k in ('start_s','end_s','wall_s','tokens_per_second','ms_per_token'):
            r[k]=float(r[k])
            if not math.isfinite(r[k]): raise ValueError('Invalid timing')
        if r['ms_per_token']<=0 or not math.isclose(r['tokens_per_second']*r['ms_per_token'],1000.,rel_tol=1e-8):
            raise ValueError('Invalid timing units')
        path=root/r['path'];summary=json.loads((path/'summary.json').read_text())
        if summary['tokens_per_second']!=r['tokens_per_second'] or sha(path/'stdout.txt')!=r['stdout_sha256']:
            raise ValueError('CSV differs from raw output')
        digests.add(r['stdout_sha256'])
        if r['kind'] not in ('warmup','anchor','reference'):
            c=specs[r['name']];cfg=json.loads((path/'config.json').read_text())
            if (tuple(cfg['cpus'])!=c.cpus or cfg['threads']!=len(c.layer)
                    or cfg['output_threads']!=len(c.output) or cfg['backend']!=c.backend
                    or cfg['output_block']!=(4 if c.row4 else 1) or not cfg['output_dynamic']):
                raise ValueError('Runtime configuration differs from planned candidate')
    if len(digests)!=1: raise ValueError('Generation mismatch')
    for name,digest in json.loads((root/'sources.json').read_text()).items():
        if sha(root/'source'/name)!=digest: raise ValueError('Source snapshot digest mismatch')
    for backend in ('openmp','pthread'):
        expected=json.loads((root/'builds'/(backend+'-sources')/'binary.json').read_text())
        if sha(root/'builds'/backend)!=expected['sha256']: raise ValueError('Built binary digest mismatch')
    measured=[r for r in rows if r['kind'] not in ('warmup','anchor')]
    if plan['args']['mode']!='topology14_search':
        expected=set(specs)|set(plan['reference_names'])
        for rep in range(1,plan['args']['repeats']+1):
            rs=[r for r in measured if r['replicate']==rep]
            if len(rs)!=len(expected) or {r['name'] for r in rs}!=expected: raise ValueError('Incomplete/duplicate block')
    else:
        trace=read_csv(root/'search_trace.csv')
        if len(trace)!=plan['args']['budget']*plan['args']['seeds']*len(plan['args']['methods']):
            raise ValueError('Incomplete search trace')
        runs={r['sequence']:r for r in measured}
        for method in plan['args']['methods']:
            for seed in range(plan['args']['seeds']):
                selected=[r for r in trace if r['method']==method and int(r['seed'])==seed]
                if [int(r['round']) for r in selected]!=list(range(1,plan['args']['budget']+1)):
                    raise ValueError('Incomplete policy rounds')
                seen={}
                for t in selected:
                    index=int(t['index']);c=space.candidate(index);r=runs[int(t['sequence'])]
                    if index in seen or c.name!=t['name'] or r['name']!=c.name or float(t['ms_per_token'])!=r['ms_per_token']:
                        raise ValueError('Trace and actual execution disagree')
                    seen[index]=r['ms_per_token']
                    if int(t['incumbent_index'])!=min(seen,key=seen.get): raise ValueError('Incumbent was not best queried point')
                if {int(r['layer_count']) for r in selected}!=set(range(1,15)): raise ValueError('Search did not cover1..14')
    dump(root/'validation.json',{'measurements':len(rows),'generation_sha256':next(iter(digests)),
         'raw_outputs_and_summaries_verified':True,'source_and_binary_hashes_verified':True,
         'runtime_configs_match_plan':True,'complete_blocks_or_search_traces_verified':True})
    return space,plan,rows,measured


def summarize(root,space,plan,rows,measured):
    specs={Candidate.from_dict(c).name:Candidate.from_dict(c) for c in plan['candidates']}
    groups={}
    for r in measured: groups.setdefault(r['name'],[]).append(r)
    cells=[]
    for name,rs in groups.items():
        c=specs.get(name)
        values=[r['ms_per_token'] for r in rs]
        cells.append({'name':name,'kind':rs[0]['kind'],'runs':len(rs),'layer_count':len(c.layer) if c else '',
                      'output_count':len(c.output) if c else '', 'backend':c.backend if c else '',
                      'dynamic':c.dynamic if c else '', 'row4':c.row4 if c else '',
                      'mean_ms_per_token':statistics.mean(values),'median_ms_per_token':statistics.median(values),
                      'mean_tokens_per_second':statistics.mean(r['tokens_per_second'] for r in rs),
                      'min_ms':min(values),'max_ms':max(values)})
    cells.sort(key=lambda r:r['mean_ms_per_token']);write_csv(root/'cells.csv',cells)
    if plan['args']['mode']=='topology14_oracle':
        curve=[]
        for n in range(1,15):
            for b in ('openmp','pthread'):
                selected=[r for r in cells if r['kind']=='fixed14' and r['layer_count']==n and r['backend']==b]
                best=min(selected,key=lambda r:r['mean_ms_per_token'])
                curve.append({'layer_count':n,'backend':b,'canonical_masks':len(selected),'best_name':best['name'],
                              'best_mean_ms':best['mean_ms_per_token'],'best_mean_tokens_per_second':best['mean_tokens_per_second']})
        write_csv(root/'depth_curve.csv',curve)
        lookup={(r['name'],r['replicate']):r for r in measured};symmetry=[]
        for alt,base in plan['symmetry_pairs'].items():
            ratios=[100*(lookup[alt,rep]['ms_per_token']/lookup[base,rep]['ms_per_token']-1) for rep in range(1,plan['args']['repeats']+1)]
            avg,lo,hi=interval(ratios)
            symmetry.append({'representative':base,'exchanged':alt,'layer_count':len(specs[base].layer),
                             'mean_latency_change_pct':avg,'ci95_low':lo,'ci95_high':hi,
                             'per_block_change_pct':json.dumps(ratios)})
        write_csv(root/'symmetry_checks.csv',symmetry)
    return cells


def replay(root,space,measured,seeds=10,budgets=(28,56,112,168)):
    allowed=space.fixed
    lookup={(r['name'],r['replicate']):r for r in measured if r['kind']=='fixed14'}
    reps=sorted({rep for _,rep in lookup})
    trace=[];results=[]
    for heldout in reps:
        train=[rep for rep in reps if rep!=heldout]
        training={i:statistics.mean(math.log(lookup[space.candidate(i).name,rep]['ms_per_token']) for rep in train) for i in allowed}
        testing={i:math.log(lookup[space.candidate(i).name,heldout]['ms_per_token']) for i in allowed}
        oracle=min(testing.values())
        full=min(training,key=training.get)
        results.append({'method':'exhaustive538','seed':-1,'heldout':heldout,'budget':len(allowed),
                        'recommended':space.candidate(full).name,'heldout_regret_pct':100*math.expm1(testing[full]-oracle),
                        'candidate_measurements':len(allowed)*len(train),'search_cpu_s':0.})
        for method in METHODS:
            for seed in range(seeds):
                before=time.process_time();policy=Policy(space,method,seed,allowed=allowed)
                for step in range(1,max(budgets)+1):
                    index=policy.ask();policy.tell(index,training[index]);best=policy.recommend()
                    regret=100*math.expm1(testing[best]-oracle)
                    trace.append({'method':method,'seed':seed,'heldout':heldout,'step':step,'index':index,
                                  'queried':space.candidate(index).name,'reason':policy.reason,
                                  'observed_log_latency':training[index],'incumbent':space.candidate(best).name,
                                  'heldout_regret_pct':regret})
                    if step in budgets:
                        results.append({'method':method,'seed':seed,'heldout':heldout,'budget':step,
                                        'recommended':space.candidate(best).name,'heldout_regret_pct':regret,
                                        'candidate_measurements':step*len(train),'search_cpu_s':time.process_time()-before})
        print(f'fixed-oracle replay: heldout{heldout}/{len(reps)} complete',flush=True)
    write_csv(root/'replay_trace.csv',trace);write_csv(root/'replay_runs.csv',results)
    summary=[]
    for method in (*METHODS,'exhaustive538'):
        for budget in (budgets if method!='exhaustive538' else (len(allowed),)):
            chosen=[r for r in results if r['method']==method and r['budget']==budget]
            values=sorted(r['heldout_regret_pct'] for r in chosen)
            summary.append({'method':method,'budget':budget,'replays':len(chosen),
                            'mean_regret_pct':statistics.mean(values),'median_regret_pct':statistics.median(values),
                            'p90_regret_pct':values[math.ceil(.9*len(values))-1],
                            'within3pct_rate':statistics.mean(v<=3 for v in values),
                            'mean_candidate_measurements':statistics.mean(r['candidate_measurements'] for r in chosen),
                            'mean_search_cpu_s':statistics.mean(r['search_cpu_s'] for r in chosen)})
    write_csv(root/'replay_summary.csv',summary)
    dump(root/'replay_protocol.json',{'space':'269 canonical fixed masks x2 backend; dynamic16/outputdynamic32,row1',
         'candidates':len(allowed),'full_joint_space_evaluated_by_oracle':False,'budgets':budgets,'seeds':list(range(seeds)),
         'training':'mean log raw ms/token over other2 blocks','evaluation':'untouched heldout1 block',
         'regret':'recommended heldout latency / minimum heldout latency over538 -1',
         'mandatory_prefix':'one queried fixed mask for each size1..14',
         'caveat':'same3 measured blocks are reused;30 replays are not30 independent experiments'})
    return summary


def shortlist(oracle,search,space,oracle_cells):
    winners=json.loads((search/'winners.json').read_text())
    specs={Candidate.from_dict(c).name:Candidate.from_dict(c) for c in json.loads((oracle/'plan.json').read_text())['candidates']}
    chosen={};sources=[]
    for w in winners:
        for rank,obj in enumerate(w['top3'],1):
            c=Candidate.from_dict(obj);chosen[c.name]=c
            sources.append({'method':w['method'],'seed':w['seed'],'rank':rank,'name':c.name})
    ranked=[r for r in oracle_cells if r['kind']=='fixed14']
    for row in ranked[:4]: chosen[row['name']]=specs[row['name']]
    refinements=[]
    for w in winners:
        refinements.append(Candidate.from_dict(w['top3'][0]))
    refinements.extend(specs[r['name']] for r in ranked[:2])
    for c in refinements:
        alt=mirror(c,space.machine)
        chosen[alt.name]=alt
    for c in (prior_best(),control(space.machine)):
        chosen[c.name]=c
    for backend in ('openmp','pthread'):
        full=next(space.candidate(i) for i in space.fixed if sum(space.layer_key(i))==14 and space.candidate(i).backend==backend)
        chosen[full.name]=full
    write_csv(search/'finalist_sources.csv',sources)
    dump(search/'selection.json',{'rule':'top3 each real search + top4 fixed oracle + mirrored top1 representatives + previous best + control + fixed14 boundaries',
         'oracle':str(oracle),'search_trace_sha256':sha(search/'search_trace.csv'),
         'oracle_trials_sha256':sha(oracle/'trials.csv'),'candidates':[asdict(c) for c in chosen.values()]})
    print(f'Independent confirmation shortlist: {len(chosen)} candidates +2 references',flush=True)


def confirmation_comparisons(root,search,cells,measured):
    candidates=[r for r in cells if r['kind']=='confirmation']
    best=min(candidates,key=lambda r:r['mean_ms_per_token'])
    lookup={(r['name'],r['replicate']):r for r in measured}
    reps=sorted({rep for _,rep in lookup});by_name={r['name']:r for r in cells}
    comparisons=[]
    for cell in cells:
        logs=[math.log(lookup[cell['name'],rep]['ms_per_token']/lookup[best['name'],rep]['ms_per_token']) for rep in reps]
        avg,lo,hi=interval(logs)
        comparisons.append({'name':cell['name'],'best_confirmed':best['name'],'best_speedup_pct':100*math.expm1(avg),
                            'ci95_low_pct':100*math.expm1(lo),'ci95_high_pct':100*math.expm1(hi),'blocks':len(reps)})
    write_csv(root/'paired_vs_best.csv',comparisons)
    methods=[]
    for w in json.loads((search/'winners.json').read_text()):
        name=Candidate.from_dict(w['top3'][0]).name;cell=by_name[name]
        methods.append({'method':w['method'],'seed':w['seed'],'queries':w['queries'],'selected':name,
                        'confirmed_mean_tokens_per_second':cell['mean_tokens_per_second'],
                        'confirmed_mean_latency_gap_pct':100*(cell['mean_ms_per_token']/best['mean_ms_per_token']-1),
                        'all_layer_sizes_1_to14':w['layer_sizes']==list(range(1,15))})
    write_csv(root/'actual_methods.csv',methods)
    return best,comparisons,methods


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
                      *['|'+'|'.join(map(str,r))+'|' for r in rows]])


def describe(c,space):
    key=space.keys[space.identify(c)//8]
    p='+'.join(str(x[0]) for x in key[:2]);e='+'.join(str(x[0]) for x in key[2:4]);lp=key[4][0]
    alt=' [交換代表]' if space.candidate(space.identify(c)).name!=c.name else ''
    return f'{c.backend} P[{p}] E[{e}] LP{lp}, {len(c.layer)}→{len(c.output)}, {"D" if c.dynamic else "S"}, {4 if c.row4 else 1}行{alt}'


def report(oracle,search,confirm,space,oracle_cells,confirm_cells,best,paired,methods):
    protocol=json.loads((oracle/'plan.json').read_text())
    ospecs={Candidate.from_dict(c).name:Candidate.from_dict(c) for c in protocol['candidates']}
    cspecs={Candidate.from_dict(c).name:Candidate.from_dict(c) for c in json.loads((confirm/'plan.json').read_text())['candidates']}
    curve=read_csv(oracle/'depth_curve.csv');sym=read_csv(oracle/'symmetry_checks.csv')
    replays=read_csv(oracle/'replay_summary.csv')
    trace=read_csv(search/'search_trace.csv');cost=[]
    for path in (oracle,search,confirm):
        rows=read_csv(path/'trials.csv')
        cost.append({'phase':path.name,'total':len(rows),'candidate_runs':sum(r['kind'] not in ('warmup','anchor') for r in rows),
                     'controls':sum(r['kind']=='anchor' for r in rows),'warmups':sum(r['kind']=='warmup' for r in rows),
                     'summed_run_wall_s':sum(float(r['wall_s']) for r in rows)})
    bestc=cspecs[best['name']]
    total=sum(r['total'] for r in cost)
    symmetry_abs=sorted(abs(float(r['mean_latency_change_pct'])) for r in sym)
    summary_methods=[]
    for method in METHODS:
        values=[r for r in methods if r['method']==method]
        summary_methods.append({'method':method,'actual_searches':len(values),
                                'mean_confirmed_gap_pct':statistics.mean(r['confirmed_mean_latency_gap_pct'] for r in values),
                                'min_gap_pct':min(r['confirmed_mean_latency_gap_pct'] for r in values),
                                'max_gap_pct':max(r['confirmed_mean_latency_gap_pct'] for r in values),
                                'total_candidate_queries':sum(r['queries'] for r in values),
                                'summed_policy_cpu_s':sum(float(r['policy_cpu_s']) for r in trace if r['method']==method)})
    write_csv(confirm/'method_summary.csv',summary_methods)
    dump(oracle/'report_metrics.json',{'best_confirmed':best,'best_candidate':asdict(bestc),'costs':cost,
         'paired':paired,'actual_methods':methods,'method_summary':summary_methods,
         'symmetry_max_abs_mean_change_pct':max(symmetry_abs),'symmetry_median_abs_mean_change_pct':statistics.median(symmetry_abs),
         'full_joint_oracle_available':False})
    dump(oracle/'best_config.json',{'candidate':asdict(bestc),'cpus':bestc.cpus,'threads':len(bestc.layer),
         'output_threads':len(bestc.output),'backend':bestc.backend,'omp_schedule':'dynamic,16' if bestc.dynamic else 'static',
         'output_dynamic':True,'output_chunk':32,'output_block':4 if bestc.row4 else 1,'p_weight':1,
         'fuse_qkv':True,'fuse_ffn':True,'confirmation_mean_tokens_per_second':best['mean_tokens_per_second']})
    texts=[
        '# 1〜14 CPUのトポロジー探索とアプローチ比較（2026-10-22）',
        f'全人数1〜14を扱い、Pなし・片側P・片側SMT・奇数E・層内LP-Eも含めて探索した。'
        f'固定mask269種類を全測定し、層内/output切替を含む118,800候補では5手法×2seedを実際に動かした。'
        f'独立8ブロックの確認での最速平均は **{describe(bestc,space)}**、'
        f'**{best["mean_tokens_per_second"]:.2f} tok/s**。'
        f'計{total}測定（controlとwarmupを含む）を保存した。',
        '## 前回から広げた範囲',
        '前回480候補はP物理2個必須、E偶数などの制限があった。今回は全14論理CPUの全ての部分集合を、'
        'SMT・L2共有関係を保つ同値類へ写像した。14人の境界だけを追加した実験ではなく、'
        '層内人数1〜14を途中の性能低下でも打ち切らずに扱う。',
        table(['対象','元の組合せ','同値化後','実際の扱い'],[
            ['固定層内/output mask','2^14−1 = 16,383','269','全代表×2 backend×3ブロックを実測'],
            ['層内⊆output の入れ子mask','3^14−2^14 = 4,766,585','14,850','全型を候補として定義し、適応探索'],
            ['入れ子mask＋kernel設定','14,850×8','118,800','5手法×2seed×112候補を実測']]),
        'P物理コアごと、Eの共有L2群ごとの(l層内人数,o出力人数)を保持し、同種群と群内CPUの交換をまとめる。'
        'PとEとLP-Eは別型。2個のPと2個のE L2群の間で交換を許すが、SMTやL2の占有分割は潰さない。'
        '元の16,383固定mask全てが269代表へ写像されることをテストした。'
        'CPU rank順列を全探索したわけではなく、CPUの同種性も性能上は近似である。',
        'kernel8設定はbackend(OpenMP/pthread)×層分配(static/dynamic16)×output1/4行。'
        'QKV/FFN融合、output dynamic32物理行、p_weight1を固定した。'
        '任意chunk、重み、バリア実装までの大域最適は対象外。'
        '[事前の計画・空間とアルゴリズム](../../src/cpuprofiler/TOPOLOGY14.md)。',
        '## 実験の費用と保存物',
        table(['実験','候補/比較','control','warmup','合計','run時間合計秒'],
              [[r['phase'],r['candidate_runs'],r['controls'],r['warmups'],r['total'],f"{r['summed_run_wall_s']:.1f}"] for r in cost]),
        'run時間はsubprocess実行時間の合計。ビルド・Pythonの候補選択・保存・集計の時間を含まない。'
        '全てstories15M、最大256steps、temperature0、seed42、同じprompt。'
        '既存のCタイマーは先頭反復を除き、sampling/decode/出力を含む。モデル読込・起動を含まない。'
        '制限256まで続けるとは限らず、前回の同じ生成は区切りtokenで222 forward呼出しだった。'
        '推論は逐次実行し、検索法/seedはラウンド内でシャッフルした。'
        '温度・周波数snapshot・負荷・電源状態・実行順、ソース、バイナリ/モデルhashを保存した。',
        '## 1〜14人の固定配置の全測定',
        '同じ融合・層dynamic16・output dynamic32・1行で全269代表×2方式を3ブロック測った。'
        '下表は各人数・方式で平均latencyが最小の代表の速度。'
        'P[a+b]は2個の物理PのSMT込み人数、E[a+b]は2個のL2群の人数。'
        'ここでの最良値は、この固定kernelの538候補の中の値である。',
        table(['人数','代表型の数','OpenMP最良tok/s','pthread最良tok/s'],
              [[n,next(r['canonical_masks'] for r in curve if int(r['layer_count'])==n),
                *[f"{float(next(r['best_mean_tokens_per_second'] for r in curve if int(r['layer_count'])==n and r['backend']==b)):.2f}"
                  for b in ('openmp','pthread')]] for n in range(1,15)]),
        '[人数別の構成名](depth_curve.csv)、[全セル](cells.csv)、[生測定](trials.csv)。'
        '最良代表の選択には同じ3ブロックを使うため、その最速値は楽観的になり得る。'
        '広い空間の探索と独立確認は別フェーズで行った。',
        '## 試した5アプローチ',
        '全手法が共通の初期14測定で1〜14人を必ず一度ずつ見る。'
        '各手法の観測はaskした候補だけで、固定oracleの速度表をオンライン探索に渡していない。',
        '- **random**: 同値類の一様探索。生CPU部分集合の一様探索とは分布が違う。\n'
        '- **beam14**: 初期14点後、深さ1〜14の各段に7候補。上位3層内配置からCPUを1個追加し、output拡張とkernelも試す。前線が枯れたreplayでは再始動する。\n'
        '- **hierarchical**: 56点まで固定配置を人数で層別探索し、有力層内配置のoutput拡張/kernelを後段で探索。早期の枝刈りに依存する。\n'
        '- **model_counts**: P/E/LPの合計人数とkernelで逐次ridge推定。SMTの物理構造とL2占有分割を予測特徴から除く。\n'
        '- **model**: SMT物理構造、L2群使用数/占有集中度、層内/output人数とkernelの相互作用を含む40特徴。'
        '両モデルはグローバル候補と最良4点のグラフ近傍から最大256点を提案し、LCBを使い、5回に1回一様探索する。',
        'model_countsも候補表現とグラフはmodelと同じ。比較しているのは予測特徴からキャッシュ/SMTの情報を除く効果である。'
        '探索指標は真の速度の信頼区間ではない。推薦は観測済みの最良候補に限る。']
    texts.extend([
        '## 固定538候補での先読みなしreplay',
        '3ブロックのleave-one-block-outで、2ブロック平均log latencyを探索観測、残り1ブロックを評価専用にした。'
        '各10seed、28/56/112/168照会。regretは評価ブロックにおける推薦latency/全538候補最小latency−1。'
        '探索器へ未照会や評価用の速度は渡さない。有限oracleは固定mask・dynamic・1行のみで、'
        '入れ子mask118,800候補の最良を知っているわけではない。',
        table(['手法','照会','平均regret%','p90%','3%以内率'],
              [[r['method'],r['budget'],f"{float(r['mean_regret_pct']):.2f}",f"{float(r['p90_regret_pct']):.2f}",
                f"{100*float(r['within3pct_rate']):.1f}%"] for r in replays]),
        '同じ3ブロックを使い回すため、30replayは30回の独立実験ではない。'
        '1照会は訓練2ブロックの2測定を平均しており、1候補1測定の実測探索と測定回数も異なる。'
        '[全照会trace](replay_trace.csv)、[fold/seed別](replay_runs.csv)、[プロトコル](replay_protocol.json)。',
        '## 118,800候補の広い空間を実際に探索',
        '5手法×seed0/1を112候補ずつ実行した。全手法で1〜14人が観測され、'
        'beam14も初期14点の後に深さ1〜14の全段を完了した。'
        '各探索の最初の推薦1点を、測定前に固定した独立8ブロックで確認した結果が次表。'
        '基準は全探索・oracle等の確認候補集合内の最速平均であり、全118,800候補の真の最適に対するregretではない。',
        table(['手法','seed','推薦構成の確認tok/s','確認最速とのlatency差%','1〜14人'],
              [[r['method'],r['seed'],f"{r['confirmed_mean_tokens_per_second']:.2f}",
                f"{r['confirmed_mean_latency_gap_pct']:.2f}",r['all_layer_sizes_1_to14']] for r in methods]),
        table(['手法','2seed平均の差%','範囲%','候補照会計','選択CPU秒計'],
              [[r['method'],f"{r['mean_confirmed_gap_pct']:.2f}",f"{r['min_gap_pct']:.2f}〜{r['max_gap_pct']:.2f}",
                r['total_candidate_queries'],f"{r['summed_policy_cpu_s']:.2f}"] for r in summary_methods]),
        '2seedだけなので方式の小差を一般化しない。予測モデルの特徴追加が必ず勝つとも、'
        '段階的な枝刈りが常に良いともみなさず、上の実測値とreplayを分けて読む。'
        f'[実測探索の全trace](../{search.name}/search_trace.csv)と'
        f'[方法別・確認値](../{confirm.name}/actual_methods.csv)を保存した。',
        '## 最終候補の独立確認',
        '各探索の上位3、固定oracle上位4、有力候補の交換代表、前回最良、control、14人固定境界を、確認前に選んだ。'
        'この中から最良平均を再選択するため、winnerの値にはなお選択バイアスが残る。'
        '以下は確認フェーズだけの速度で、oracleフェーズの絶対速度と混ぜない。',
        table(['候補','tok/s','ms/token'],
              [[describe(cspecs[r['name']],space) if r['name'] in cspecs else r['name'],
                f"{r['mean_tokens_per_second']:.2f}",f"{r['mean_ms_per_token']:.4f}"] for r in confirm_cells]),
        '最速平均の設定は [best_config.json](best_config.json)。'
        'ソースと設定の比較であり、pthread/OpenMP一般の優劣や単一の物理的原因に帰属しない。'])
    for name,label in (('upstream','未改造upstream'),('qkvffn_branch','未改造qkv-ffn'),(prior_best().name,'前回の採用候補')):
        comparison=next(r for r in paired if r['name']==name)
        texts.append(f'{label}に対する最速候補の速度改善は、ブロック内latency比の幾何平均で'
                     f' {comparison["best_speedup_pct"]:+.2f}%、探索的95%区間'
                     f' [{comparison["ci95_low_pct"]:+.2f}, {comparison["ci95_high_pct"]:+.2f}]%。'
                     +('ゼロをまたぐため、この差は未確定。' if comparison['ci95_low_pct']<=0<=comparison['ci95_high_pct'] else ''))
    texts.extend([
        '## 同値化ヒューリスティックの点検と限界',
        'P物理コア/EのL2群/LP-Eを交換した28配置を、代表と同じ条件で3ブロックずつ測った。'
        f'交換による平均latency変化の絶対値は中央値{statistics.median(symmetry_abs):.2f}%、'
        f'最大{max(symmetry_abs):.2f}%だった。'
        'これは同じ種類でも常に同速とは限らないことと、短い測定の雑音の両方を含む。'
        '温度・負荷・周波数snapshotだけでは原因を区別できない。',
        '[交換28対の結果](symmetry_checks.csv)には3ブロックの各差と探索的区間を保存した。'
        '同値化は探索費用を減らすための近似である。真の最適CPU番号やrank順列の保証はない。'
        'このため、各探索の上位1と固定oracle上位2は交換代表も独立確認に含めた。'
        '全同値類の全交換を調べたものではなく、残る代表やCPU順序は今後の検証点となる。',
        '今回測った固定oracleは538候補であり、全118,800候補を総当たりしたとは主張しない。'
        '実測探索の112候補は全体の約0.094%だが、初期点・control・再確認の費用を含めて評価する必要がある。'
        '前回の粗いcontrol補正での失敗を踏まえ、今回は生値を目的にした。'
        'それでも一時的に速い点を選ぶ危険があり、独立確認を必須の評価手順とした。'
        '大型モデル・長文・別CPU・並行要求での最適性、任意chunkやバリア方式の最適性は未検証。',
        '## 再実行と検証',
        '標準ライブラリのみで測定・探索・集計を行える。'
        '[TOPOLOGY14.md](../../src/cpuprofiler/TOPOLOGY14.md)のコマンドで新規フォルダへ再実行できる。'
        'source/に測定時のソース、analysis_source/に集計時のPythonを保存した。'
        '全stdout hash、summary/CSV、実行設定、ソース/バイナリhash、完全ブロック、'
        '探索traceと全人数到達を各validation.jsonで確認した。'
        '固定16,383maskの同値化、14,850入れ子型の往復、特徴のアブレーション、'
        '深さ14到達、先読みなし選択、ridge更新もテストした。'])
    (oracle/'README.md').write_text('\n\n'.join(texts)+'\n')
    dump(oracle/'report_provenance.json',{'script_sha256':sha(Path(__file__)),
         'inputs':{str(path):sha(path) for path in (oracle/'trials.csv',oracle/'replay_summary.csv',
             search/'search_trace.csv',confirm/'trials.csv',confirm/'cells.csv')}})
    print(f'Report: {oracle/"README.md"}',flush=True)


def snapshot_analysis(root):
    hashes={}
    for source in Path(__file__).parent.glob('*.py'):
        dest=root/'analysis_source'/'src'/'cpuprofiler'/source.name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        hashes[source.name]=sha(source)
    dump(root/'analysis_provenance.json',{'source_sha256':hashes,'trials_sha256':sha(root/'trials.csv')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--search',type=Path)
    parser.add_argument('--confirm',type=Path)
    parser.add_argument('--no-replay',action='store_true')
    args=parser.parse_args()
    space,plan,rows,measured=validate(args.oracle);snapshot_analysis(args.oracle)
    cells=summarize(args.oracle,space,plan,rows,measured)
    if not args.no_replay: replay(args.oracle,space,measured)
    if not args.search: return
    s,sp,srows,smeasured=validate(args.search);snapshot_analysis(args.search)
    summarize(args.search,s,sp,srows,smeasured)
    if not args.confirm:
        shortlist(args.oracle,args.search,space,cells)
        return
    c,cp,crows,cmeasured=validate(args.confirm);snapshot_analysis(args.confirm)
    ccells=summarize(args.confirm,c,cp,crows,cmeasured)
    best,paired,methods=confirmation_comparisons(args.confirm,args.search,ccells,cmeasured)
    report(args.oracle,args.search,args.confirm,space,cells,ccells,best,paired,methods)


if __name__=='__main__': main()
