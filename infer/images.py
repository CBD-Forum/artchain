#!/usr/bin/python
#-*- encoding:utf-8 -*-
import tornado.ioloop
import tornado.web
import os
import tornado.ioloop
import tornado.web
import signal
import logging
from tornado.options import options
from infer_eval import evaluate

from PIL import Image
import numpy as np

def process(filepath):
    return evaluate(filepath)

class UploadFileHandler(tornado.web.RequestHandler):
    def get(self):
        pass

    def post(self):
        upload_path=os.path.join(os.path.dirname(__file__),'./pictues')  #文件的暂存路径
        file_metas=self.request.files['file']    #提取表单中‘name’为‘file’的文件元数据
        for meta in file_metas:
            filename=meta['filename']
            filepath=os.path.join(upload_path,filename)
            with open(filepath,'wb') as up:
                up.write(meta['body'])
        binfile = pic2bin(resize(filepath))  # 处理文件
        dpResult = process(binfile)
        print(dpResult)
        self.write(str(dpResult) )


# http request holder
app=tornado.web.Application([
    (r'/images',UploadFileHandler),
])

def resize(filepath):
    width = 32
    height = 32
    im = Image.open(filepath)
    out = im.resize((width, height), Image.ANTIALIAS)
    resized = filepath
    out.save(resized)
    return resized

def pic2bin(filepath):
    bin_path = os.path.join(os.path.dirname(__file__), './pic2bin')
    filename = filepath
    im = Image.open(filename)
    im = (np.array(im))

    r = im[:, :, 0].flatten()
    g = im[:, :, 1].flatten()
    b = im[:, :, 2].flatten()
    label = [1]

    out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    binfile = bin_path + '/'+os.path.basename(filepath)+".bin"
    print(binfile)
    out.tofile(binfile)
    return binfile

is_closing = False

def signal_handler(signum, frame):
    global is_closing
    logging.info('exiting...')
    is_closing = True

def try_exit():
    global is_closing
    if is_closing:
        # clean up here
        tornado.ioloop.IOLoop.instance().stop()
        logging.info('exit success')

if __name__ == '__main__':
    tornado.options.parse_command_line()
    signal.signal(signal.SIGINT, signal_handler)
    app.listen(3000)
    tornado.ioloop.PeriodicCallback(try_exit, 100).start()
    tornado.ioloop.IOLoop.instance().start()