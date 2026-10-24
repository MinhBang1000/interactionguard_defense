import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_runner.plots import render


class PlotTests(unittest.TestCase):
    def test_empty_summary_requires_no_plotting_dependency(self):
        self.assertEqual(render({}, '.' )['created'], [])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional matplotlib is not installed')
    def test_cpu_sweep_plot_is_created_in_temporary_directory(self):
        summary = {}
        for name, latency in [('benign_90', .4), ('benign_98', .15)]:
            summary[name] = {'trials': 5, 'metrics': {
                'f1': {'mean': .88, 'std': .01},
                'estimated_pipeline_seconds_per_sample': {'mean': latency, 'std': .02}},
                'traffic': {layer: {'mean': .05, 'std': .01} for layer in ('Gate', 'Screen', 'Audit')}}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'MPLCONFIGDIR': temp}):
            output = render(summary, temp)
            self.assertEqual(output['created'], ['comparison.png'])
            self.assertGreater((Path(temp) / 'comparison.png').stat().st_size, 1000)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional matplotlib is not installed')
    def test_paired_benign_comparison_and_per_order_traffic_plots(self):
        summary = {}
        for order in ('Gate-Screen-Audit', 'Screen-Gate-Audit'):
            for ratio in (90, 98):
                summary[f'benign_{ratio}/{order}'] = {'trials': 5, 'metrics': {
                    'f1': {'mean': .85, 'std': .02},
                    'estimated_pipeline_seconds_per_sample': {'mean': .2, 'std': .01}},
                    'traffic': {layer: {'mean': .1, 'std': .01} for layer in ('Gate', 'Screen', 'Audit')}}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'MPLCONFIGDIR': temp}):
            result = render(summary, temp)
            self.assertEqual(set(result['created']), {'comparison.png',
                'Gate-Screen-Audit/comparison.png', 'Screen-Gate-Audit/comparison.png'})
            for name in result['created']:
                self.assertGreater((Path(temp) / name).stat().st_size, 1000)
            summary.pop('benign_98/Screen-Gate-Audit')
            with self.assertRaisesRegex(ValueError, 'every configuration'):
                render(summary, temp)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional matplotlib is not installed')
    def test_threshold_orders_keep_traffic_charts_labels_and_no_titles(self):
        from matplotlib.figure import Figure
        summary = {f'{point}/{order}': {'trials': 5, 'metrics': {
            'f1': {'mean': .88, 'std': .01},
            'estimated_pipeline_seconds_per_sample': {'mean': .15, 'std': .02}},
            'traffic': {layer: {'mean': .05, 'std': .01} for layer in ('Screen', 'Gate', 'Audit')}}
            for point in ('validation_max_f1', 'recall_0.999')
            for order in ('Screen-Gate-Audit', 'Gate-Screen-Audit')}
        original = Figure.savefig
        captured = []
        def inspect(figure, *args, **kwargs):
            self.assertTrue(all(not axes.get_title() for axes in figure.axes))
            self.assertEqual(len(figure.legends), 1)
            captured.append([t.get_text() for t in figure.legends[0].get_texts()])
            return original(figure, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'MPLCONFIGDIR': temp}), \
                patch.object(Figure, 'savefig', inspect):
            result = render(summary, temp)
            self.assertEqual(set(result['created']), {'comparison.png',
                'Screen-Gate-Audit/comparison.png', 'Gate-Screen-Audit/comparison.png'})
            self.assertTrue(all((Path(temp) / name).stat().st_size > 1000 for name in result['created']))
            self.assertIn('Samples processed by Gate', captured[0])
            self.assertIn('Screen-Gate-Audit: F1', captured[-1])
            self.assertIn('Gate-Screen-Audit: F1', captured[-1])
            summary.pop('recall_0.999/Screen-Gate-Audit')
            with self.assertRaisesRegex(ValueError, 'every configuration'):
                render(summary, temp)


if __name__ == '__main__':
    unittest.main()
