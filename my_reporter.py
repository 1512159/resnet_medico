""" reporter.py

    This is intended a as a simple drop-in evaluation analyzer for tensorflow.

    You can create a `tf_reporter` layer which accepts the following tensors:
    1. image:     [batch x W x H x 3]
    2. predicted: [batch]
    3. expected:  [batch]

    The `tf_reporter` is abstracted within the `EvalReporter` class which can
    be queried for the tf `op` which can be inserted into the evaluation
    session's execution graph.

    TODO: Detailed usage etc.

"""
import scipy
import base64
import cStringIO
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from jinja2 import Template
import os
CLASSES = (
    'nothing',
    'colon',
    'lifted-plp',
    'resection',
    'esopha',
    'instr',
    'cecum',
    'pylorus',
    'z-line',
    'outside',
    'polyps',
    're-rectum',
    're-stomach',
    'stool-icl',
    'stool-ple',
    'ulcerative',
)
################################################################################

class EvalReporter(object):
    """ EvalReporter is a class which is constructed with a given batches set of
        tensors for the image, predictions and the expectation.
    """

    failure_histogram = None        # incorrectly predicted classes
    success_histogram = None        # correctly predicted classes
    op                = None        # `py_func` layer which dumps images


    def __init__(self, img_ids=[], images=[], predicted=[], expected=[], prob=[]):
        """ constructor for the evaluation reporter.
            TODO: The lengths of the incoming tensors must match!
        """
        self.clear()
        self.op = tf.py_func(self._pyfunc, [img_ids, images, predicted, expected, prob], tf.float32)


    def _pyfunc(self, img_ids, images, predicted, expected, prob):
        """ _pyfunc is the python_func's "op" which will accept a list of the
            images, predicted classes and the expectations.  These at this point
            are NOT tensors, but are `numpy.ndarray`'s.
        """
        total = 0
        error = 0
        for i in range(len(images)):
            im = scipy.misc.toimage(images[i])              # ndarray -> PIL.Image
            #calculate prob dict
            prob_dict = {}
            prob_list = prob[i]
            for k in xrange(len(prob_list)):
                prob_dict[CLASSES[k]] = round(prob_list[k],3)

            #write prob
            d = ImageDraw.Draw(im)
            fnt =ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18, encoding="unic")
            # d.rectangle(((0,0),(200,150)), fill = "white")
            sorted_d = sorted(prob_dict.items(), key=lambda x: x[1], reverse = True)
            d.text((20, 0), os.path.basename(img_ids[i]), font=fnt, fill=(255,255,255,255))
            for k, item in enumerate(sorted_d[:7]):
                d.text((20, 20 + k * 20), '{:<12}:{:.3f}'.format(str(item[0]), float(item[1])), font=fnt, fill=(0,255,0,255))
            
            bs = cStringIO.StringIO()                       # buffer to hold image
            im.save(bs, format="JPEG")                      # PIL.Image -> JPEG
            b64s = base64.b64encode(bs.getvalue())          # JPEG -> Base64
            total += 1
           
            if CLASSES[expected[i]] == CLASSES[predicted[i]]:
                if CLASSES[predicted[i]] in self.success_histogram:
                    self.success_histogram[CLASSES[predicted[i]]].append((b64s, prob_dict))
                else:
                    self.success_histogram[CLASSES[predicted[i]]] = [(b64s, prob_dict)]
            else:
                error += 1
                if CLASSES[predicted[i]] in self.failure_histogram:
                    self.failure_histogram[CLASSES[predicted[i]]].append([b64s, CLASSES[expected[i]], prob_dict])
                else:
                    self.failure_histogram[CLASSES[predicted[i]]] = [[b64s, CLASSES[expected[i]], prob_dict]]

        return np.float32(((total - error)/total) if total > 0 else 0)


    def get_op(self):
        """ get_op returns the tensorflow wrapped `py_func` which will convert the local
            tensors into numpy ndarrays.
        """
        return self.op


    def clear(self):
        """ clear resets the histograms for `this` reporter.
        """
        self.failure_histogram = dict()
        self.success_histogram = dict()


    def write_html_file(self, file_path):
        """ write_html_file dumps the current histograms to the specified `file_path`.
        """
        report = Template("""<!DOCTYPE html>
<html>
<head>
    <title>Prediction Analyzer</title>
</head>
<body>
    <h1>Correct Predictions</h1><br>
    {%  for class, images in me.success_histogram.items() %}
        <h3>Class {{ class }}</h3><br>
        {%  for img in images %}
            <img src="data:image/jpeg;base64,{{img[0]}}" title="class:{{class}} {{img[1]}}" />
        {%  endfor %}
    {%  endfor %}
    <hr><br><br>

    <h1>Incorrect Predictions</h1><br>
    {%  for class, groups in me.failure_histogram.items() %}
        <h3>Class {{ class }}</h3><br>
        {%  for group in groups %}
            <img src="data:image/jpeg;base64,{{group[0]}}" title="pred:{{class}} exp:{{group[1]}} prob:{{group[2]}}" />
        {%  endfor %}
    {%  endfor %}
    <hr><br><br>
</body>
</html>
""").render(me=self)
        with open(file_path, "w") as fout:
            fout.write(report)

################################################################################
