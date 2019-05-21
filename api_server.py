from flask import Flask, render_template, request
import os
import darknet_detect
import numpy as np
import sys
import json
import collections
import time
import base64
import io
from PIL import Image
from config import DETECT_THRESHOLD

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/api/predict/object", methods=['POST'])
def ai_object():
    upload_image = request.form['image']
    image = base64.b64decode(upload_image)
    image = np.fromstring(image, np.uint8)
    image = io.BytesIO(image)
    image = Image.open(image).convert('RGB')

    t = time.time()
    classes, boxes, scores = darknet_detect.detect(image)

    format_result = collections.OrderedDict()
    results = []
    
    print(scores)

    if len(classes) > 0:
        for i in range(len(scores)):
            if scores[i] < DETECT_THRESHOLD:
                break
            location = collections.OrderedDict()

            xmin = max(boxes[i][0], 0)
            ymin = max(boxes[i][1], 0)
            xmax = min(boxes[i][2], image.size[0])
            ymax = min(boxes[i][3], image.size[1])
            w = xmax - xmin
            h = ymax - ymin

            location['xmin'] = xmin
            location['ymin'] = ymin
            location['w'] = w
            location['h'] = h

            result = collections.OrderedDict()
            result['name'] = classes[i]
            result['score'] = scores[i]
            result['location'] = location
            
            results.append(result)

    cost_time = round(time.time()-t, 3)
    format_result["res"] = results
    format_result["num"] = len(results)
    format_result["time"] = cost_time

    print("done. it tooks {:.3f}s.".format(time.time()-t))
    # return json.dumps(format_result, cls=JsonEncoder, ensure_ascii=False)
    return json.dumps(format_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 4321)
