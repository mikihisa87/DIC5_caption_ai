# TFRecord形式での書き込み
-*- coding:utf-8 -*-

import glob
import tensorflow as tf

# jpegファイルのパスを取得
img_list = [i for i in glob.glob('img/*.jpg')]

with tf.python_io.TFRecordWriter('test.tfrecord') as w:

    for img in img_list:

        #ファイルをバイナリとして読み込み
        with tf.gfile.FastGFile(img, 'rb') as f:
            data = f.read()

#取得したbyte列をkey.valueに登録
features = tf.train.Features(feature={
    'data':tf.train.Feature(bytes_list=tf.train.BytesList(Values=[data]))
})

#Exampleクラスにkey, valueを登録して書き込み
example = tf.train.Example(features=features)
w.write(example.SerializeToString())

# TFRecord形式ファイルの読み込み
-*- coding:utf-8 -*-

import tensorflow as tf
from skimage import io

#TFRecordファイルを読み込みパース用の関数を適用
fataset = tf.data.TFRecordDataset(['test.tfrecord']).map(parse)

#TFRecord形式のパース処理
def parse(example):

    #TFRecordをパース
    features = tf.parse_single_example(
        example,
        features={
            'data': tf.FixedFeature([], dtype=tf.string)
        }
    )
    #バイト列のママになっているので、元の画像に変換
    img = features['data']
    img = tf.image.decode_jpeg(img)
    return img

    #データセットを１週するイテレータ
    iterator = dataset.make_one_shot_iterator()
    #イテレータから要素を取得
    next_element = iterator.get_next()

    with tf.Session() as sess:
        #データセットから画像を１件取得
        jpeg_img = sess.run(next_element)
        #scikit-imageで表示
        io.imshow(jpeg_img)
        io.show()

# モジュールインポート
-*- coding:utf-8 -*-

import os
import json
import numpy as np
from collections import namedtuple, Counter
import tensorflow as tf

tf.flags.DEFINE_string('train_img_dir', 'data/img/train2014', 'Training image directory.')
tf.flags.DEFINE_string('val_img_dir', 'data/img/val2014', 'validation image directory.')
tf.flags.DEFINE_string('train_captions', 'data/stair_captions_v1.1_train.json', 'Training caption file.')
tf.flags.DEFINE_string('val_captions', 'data/stair_captions_v1.1_val.json', 'Validation caption file.')
tf.flags.DEFINE_string('out_dir', 'data/tfrecords/', 'Output TFRecords directory.')
tf.flags.DEFINE_integer('min_word_count', 4, 'The minimum number of occurrences of each word in the training set for includion in the vocab.')
tf.flags.DEFINE_string('word_list_file', 'data/dictinary.txt', 'Output word list file.')

FLAGS = tf.flags.FLAGS

START_WORD = '<S>'
END_WORD = '<E>'
UNKNOWN_WORD = '<UNW>'

NUM_TRAIN_FILE = 256
NUM_VAL_FILE = 4
NUM_TEST_FILE = 8

ImageMetadata = namedtuple('ImageMetadata', ['img_id', 'filename'])

#Jsonファイルを読み込み画像のid、ファイル名、キャプションを取得する。
def _load_metadata(caption_filename, img_dir):

    #jsonファイルをロード
    with open(caption_filename, 'r') as f:
        meta_data = json.load(f)

    #画像idとファイル名を持つnamedtupleのリストを作成
    meta_list = [ImageMetadata(x['id'], os.path.join(img_dir, x['file_name'])) for x in meta_data['images']]

    #スペース区切りのcaptinを単語の配列に変換
    def _create_word_list(caption):
        tokenized_captions = [START_WORD]
        tokenized_captions.extend(captin.split())
        tokenized_captions.append(END_WORD)
        return tokenized_captions

    #{画像id => キャプションのリスト}の辞書を作成
    id_to_captions = {}
    for annotatin in meta_data['annotations']:
        img_id = annotation['image_id']
        caption = annotation['tokenized_captions']
        caption = _create_word_list(caption)

    print('Loaded caption metadata for %d images from %s' %
            len(meta_list), caption_filename)

def main(argv):

    #jsonファイルからメタデータの読み込み
    #(画像id、ファイルパス)のタプるの配列{id=>キャプションのリスト}を取得
    train_meta, train_captions = _load_metadata(FLAGS.train_captions, FLAGS.train_img_dir)
    val_meta, val_captions = _load_metadata(FLAGS.val_captions, FLAGS.val_img_dir)

    #キャプションをマージ
    captins = {k:v for dic in [train_captions, val_captions]
            for k, v in dic.items()}

    #訓練データ、バリデーションデータ、テストデータに分割
    train_cutoff = int(0.85 * len(val_meta))
    val_cutoff = int(0.90 * len(val_meta))

    train_dataset = train_meta + val_meta[0:train_cutoff]
    val_dataset = val_meta[train_cutoff:val_cutoff]
    test_dataset = val_meta[val_cutoff:]

    #訓練データから辞書作成
    train_captions = []
    for meta in train_dataset:
        c = captions[meta.img_id]
        train_captions.append(c)

    word_to_id, id_to_word = _create_vocab(train_captions)

    def _create_vocab(captions):

        counter = Counter()
        for c in captins:
            counter.update(c)
        print('total words:', len(counter))
        #出現回数が一定数のものだけ辞書に採用。出現回数抗降順でソート
        #word_countsは（単語。出現回数）のリスト
        word_counts = [x for x in counter.items()
            if x[1] >= FLAGS.min_word_count]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        print('Words in vacab:', len(word_counts))

        #辞書作成
        word_list = [x[0] for x in word_counts]
        #<S>と<E>のidを1,0で固定したいので、一度削除して先頭に追加する
        word_list.remove(START_WORD)
        word_list.remove(END_WORD)
        word_list.insert(0, START_WORD)
        word_list.insert(0, END_WORD)

        word_list.append(UNKNOWN_WORD)
        word_to_id = dict([(x, y) for (y, x) in enumerate(word_list)])
        id_to_word = dict([(x, y) for (x, y) in enumerate(word_list)])
        return word_to_id, id_to_word


    #画像を読み込みメタデータと結合したバイナリを作成
    _create_datasets('train', train_dataset, captions,
            word_to_id, NUM_TRAIN_FILE)
    _create_datasets('val', val_dataset, captions, word_to_id, NUM_VAL_FILE)
    _create_datasets('test', test_dataset, captions, word_to_id, NUM_TEST_FILE)

    #単語リスト出力
    with open(FLAGS.word_list_file, 'a') as f:
        for k, v in id_to_word.items():
            f.write(v)
            f.write('\n')

    #画像メタデータと辞書を元に指定されたファイル数に分割してTFRecordを作成する
    def _create_datasets(name, img_meta, captions, word_to_id, num_file):

        #画像メタデータをだいたい等しく分割
        img_chunk = np.array_split(img_meta, num_file)
        counter = 0
        for i in range(1, num_file + 1):
            output_file_name = '%s-%.3d.tfrecord' % (name, i)
            output_file_path = os.path.join(FLAGS.out_dir, output_file_name)
            target_chunk = img_chunk[counter]

        # 対象画像群書ごとにWriterを定義
        with tf.python_io.TFRecordWriter(output_file_path) as writer:
            for img in target_chunk:
                img_id = img[0]
                filename = img[1]
                #画像ファイルをバイト列として読み込み
                with tf.gfile.FastGFile(filename, 'rb') as f:
                    data = f.read()

                #キャプションのid化
                caption = captions[int(img_id)]
                caption_ids = []
                for w in caption:
                    if w in word_to_id:
                        caption_ids.append(word_to_id[w])
                    else:
                        caption_ids.append(word_to_id[UNKNOWN_WORD])

                #固定長部分
                context = tf.train.Features(feature={
                        'img_id': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(img_id)])),
                        'data': tf.train.Feature(
                                bytes_list = tf.train.BytesList(value=[data])),
                        })

                #可変長部分
                caption_feature = [tf.train.Feature(
                        int64_list = tf.train.Int64List(value=[v])) for v in caption_ids]
                feature_lists = tf.train.FeatureList(feature_list = {
                                'caption': tf.train.FeatureList(feture=caption_feature)
                })

                # TFRecordに書き込み
                sequence_example = tf.train.SequenceExample(context=context,
                        feature_lists=feature_lists)
                writer.write(sequence_example.SerializeToString())
        counter += 1
