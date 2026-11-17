"""Real local RapidOCR and direction/coordinate contracts, including refusals."""
import base64
import io
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'server'))
from ocr import TextRecognizer, select


def fixture():
    image = Image.new('RGB', (900, 600), '#eeeeee')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', 52)
    for text, point in [('确认', (100, 100)), ('确认', (620, 100)), ('Cancel', (600, 400)), ('开始探索', (100, 400))]:
        draw.text(point, text, fill='#111111', font=font)
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


class OcrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = TextRecognizer()
        cls.image = fixture()

    def infer(self, **args):
        return self.engine.infer({'image': self.image, **args})

    def test_chinese_duplicates_refused_and_direction_resolves(self):
        ambiguous = self.infer(text='确认')
        self.assertEqual(ambiguous['status'], 'ambiguous')
        self.assertIsNone(ambiguous['selected'])
        right = select(ambiguous, {'text': '确认', 'direction': 'right'})
        self.assertEqual(right['status'], 'selected')
        self.assertGreater(right['selected']['center'][0], 600)
        self.assertLess(right['selected']['center'][1], 200)
        left = select(ambiguous, {'text': '确认', 'direction': 'upper_left'})
        self.assertLess(left['selected']['center'][0], 250)
        self.assertEqual(select(ambiguous, {'text': '确认', 'occurrence': 5})['status'], 'index_out_of_range')

    def test_substring_box_is_not_whole_line(self):
        whole = self.infer(text='开始探索')['selected']
        substring = self.infer(text='探索')['selected']
        self.assertGreater(substring['box'][0], whole['box'][0]+20)
        self.assertLess(substring['box'][2]-substring['box'][0], whole['box'][2]-whole['box'][0])

    def test_english_and_y_down(self):
        result = self.infer(text='cancel', direction='lower_right')
        self.assertEqual(result['status'], 'selected')
        self.assertGreater(result['selected']['center'][1], 390)
        self.assertEqual(result['coordinateFrame'], 'image-pixels-top-left-y-down')
        self.assertEqual(self.infer(text='不存在')['status'], 'not_found')

    def test_invalid_inputs(self):
        for args in [{'image': '!!!'}, {'direction': 'behind'}, {'anchor': [float('nan'), 0]},
                     {'occurrence': True}, {'occurrence': -1}, {'text': '\x00'}, {'minConfidence': 2}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.engine.infer({'image': self.image, **args})


if __name__ == '__main__':
    unittest.main()
