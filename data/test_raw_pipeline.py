"""Real temporary-file integration: raw -> outcome -> split -> prefix -> training."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest

from data.annotate_raw import annotate_sources
from data.data_split import SOURCES, main as split
from data.data_preprocess import normalize_trace, read_jsonl, write_json, write_jsonl, main as prefixes
from data.prepare_training import prepare, load_prepared
from data.test_preprocessing import episode
from eval.asr import episode_hash, original_outcome
from eval.prompt_success_judge import reusable_record, apply_reviews


class RawPipelineTests(unittest.TestCase):
    def test_end_to_end_without_models_preserves_raw_and_prefix_rules(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root=Path(folder); members={s:[] for s in ["train","val","test"]};before={};judges=[]
            for filename,kind in SOURCES.items():
                rows=[]
                for i,s in enumerate(members):
                    t=episode(kind);t['id']=f"{kind or 'benign'}_{i}";t['prompt']=f"Question {kind} {i}?";t['label']=int(kind is not None)
                    if kind is None:
                        t['reasoning_steps']=[{'node':'retrieve','content':'ordinary facts'},{'node':'agent','content':'ordinary answer'}]
                        t['trace']=['retrieve','agent'];t['final_answer']='ordinary answer'
                    if kind=='prompt':
                        judges.append({'id':t['id'],'episode_sha256':episode_hash(normalize_trace(t)),
                                       'verdict':'FAILURE','first_success_step':None,'reason':'fixture'})
                    rows.append(t);members[s].append(t['id'])
                path=root/'raw'/filename;write_jsonl(path,rows);before[path]=path.read_bytes()
            judge_path=root/'judges.jsonl';write_jsonl(judge_path,judges)
            write_json(root/'splits/split_manifest.json',{'episode_ids':members})
            annotate_sources(root,root/'annotated',judge_path)
            split(['--data-dir',str(root),'--source-dir',str(root/'annotated'),'--require-outcomes'])
            prefixes(['--split-dir',str(root/'splits'),'--output-dir',str(root/'processed')])
            prepare(root/'processed',root/'training')
            for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)
            for s in members:
                traces=read_jsonl(root/f'splits/{s}.jsonl')
                self.assertEqual([t['id'] for t in traces],members[s])
                for t in traces:
                    self.assertIn('asr_outcome',t);original_outcome(t)
                for row in read_jsonl(root/f'processed/{s}_processed.jsonl'):
                    self.assertEqual(row['label'],int(row['first_injection_step'] is not None and row['step_number']>=row['first_injection_step']))
                    self.assertNotIn('asr_outcome',row['sequence'])
            for stage in ['encoder','autoencoder']:
                for s in ['train','val']:
                    rows=load_prepared(stage,s,output_dir=root/'training')
                    self.assertTrue(all(r['attack_type']!='correlated' for r in rows))
                    if stage=='autoencoder':self.assertTrue(all(r['label']==0 for r in rows))
            changed=root/'annotated/prompt_injection_traces.jsonl'
            altered=read_jsonl(changed);altered[0]['prompt']='tampered source'
            write_jsonl(changed,altered)
            with self.assertRaises(ValueError):
                split(['--data-dir',str(root),'--source-dir',str(root/'annotated'),'--require-outcomes'])

    def test_accepted_legacy_judge_is_reused_even_with_different_model(self):
        t=normalize_trace(episode('prompt'))
        record={'episode_sha256':episode_hash(t),'verdict':'SUCCESS','model':'gpt-4o',
                'prompt_version':'legacy_success_judge_user_accepted_20261021'}
        self.assertTrue(reusable_record(t,record,'meta-llama/Llama-3.1-8B-Instruct'))
        t['prompt']='changed'
        self.assertFalse(reusable_record(t,record,'meta-llama/Llama-3.1-8B-Instruct'))

    def test_review_preserves_llama_proposal_and_rejects_stale_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            t=normalize_trace(episode('prompt'))
            cache={t['id']:{'id':t['id'],'episode_sha256':episode_hash(t),
                           'verdict':'SUCCESS','reason':'Incorrect proposal'}}
            review={'id':t['id'],'episode_sha256':episode_hash(t),'verdict':'FAILURE',
                    'first_success_step':None,'evidence':'','reason':'Observed refusal'}
            path=Path(folder)/'reviews.jsonl';write_jsonl(path,[review])
            apply_reviews([t],cache,path)
            self.assertEqual(cache[t['id']]['verdict'],'FAILURE')
            self.assertEqual(cache[t['id']]['llama_proposal']['verdict'],'SUCCESS')
            bad={**review,'verdict':'SUCCESS','first_success_step':2,'evidence':'invented'}
            write_jsonl(path,[bad])
            with self.assertRaises(ValueError):apply_reviews([t],cache,path)


if __name__=='__main__':unittest.main()
