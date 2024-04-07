import warnings
import pickle
import argparse
import os
import time
import datetime
import logging

warnings.filterwarnings("ignore")

import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from scipy import stats
from model import HybridFeatures
from core.utils.biqa import load_data, augment, jpeg_dct
import matplotlib.pyplot as plt


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and testing Green-BIQA',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_predict', action='store_true', help='Perform prediction on images in the data directory')

    parser.add_argument('--yuv', action='store_true')
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--num_aug', type=int, default=4)

    return parser.parse_args(args)


def set_logger(args):
    log_file = os.path.join(args.output_dir, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    

def main(args):
    if args.yuv:
        assert (args.height != 0) and (args.width != 0), "Please specify height and width when using yuv."

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_logger(args)

    logging.info('Loading images from {}...'.format(args.data_dir))
    t = time.time()

    Y_feature_extractor = HybridFeatures(channel='Y')
    U_feature_extractor = HybridFeatures(channel='U')
    V_feature_extractor = HybridFeatures(channel='V')

    if args.do_train:
        start_time = time.time()
        logging.info('Start training the feature extractor...')
        mos = []
        aug_mos_list = []
        features_list = []
        num_batches = 0
        for batch_images, batch_mos in load_data(args, batch_size=64):
            num_batches += 1
            logging.info('Processing batch of size {}...'.format(len(batch_images)))
            mos.extend(batch_mos)
            batch_images, aug_mos = augment(batch_images, batch_mos, args.num_aug)
            aug_mos_list.extend(aug_mos)
            train_Y_images_block_dct = jpeg_dct(batch_images, 'Y')
            train_U_images_block_dct = jpeg_dct(batch_images, 'U')
            train_V_images_block_dct = jpeg_dct(batch_images, 'V')
            y_features = Y_feature_extractor.fit_transform(train_Y_images_block_dct, aug_mos)
            u_features = U_feature_extractor.fit_transform(train_U_images_block_dct, aug_mos)
            v_features = V_feature_extractor.fit_transform(train_V_images_block_dct, aug_mos)
            features = np.concatenate([y_features, u_features, v_features], axis=-1)
            logging.info('Feature Size = {}'.format(features.shape))
            if args.save:
                logging.info('Saving extracted features in {}'.format(args.output_dir))
                pickle.dump(features, open(os.path.join(args.output_dir, "features.pickle"), "wb"))
            features_list.append(features)
            
            # Clear the memory after processing each batch
            del batch_images, aug_mos, train_Y_images_block_dct, train_U_images_block_dct, train_V_images_block_dct
            del y_features, u_features, v_features, features
        
        features = np.concatenate(features_list, axis=0)
        aug_mos = np.array(aug_mos_list)

        n_train = int(0.9 * len(mos)) * args.num_aug
        all_index = np.arange(features.shape[0])
        train_index = all_index[:n_train]
        valid_index = all_index[n_train:]

        X_train, X_valid = features[train_index], features[valid_index]
        y_train, y_valid = aug_mos[train_index], aug_mos[valid_index]

        eval_set = [(X_train, y_train), (X_valid, y_valid)]

        reg = xgb.XGBRegressor(objective='reg:squarederror',
                            max_depth=5,
                            n_estimators=1500,
                            subsample=0.6,
                            eta=0.08,
                            colsample_bytree=0.4,
                            min_child_weight=4)

        logging.info('Start training the regressor...')
        t = time.time()
        reg.fit(X_train, y_train, eval_set=eval_set, eval_metric=['rmse'],
                early_stopping_rounds=100, verbose=False)
        logging.info('Regressor trained in {} secs...'.format(time.time() - t))

        bst = reg.get_booster()
        bst.save_model(os.path.join(args.model_dir, 'xgboost.json'))
        
        logging.info('Validating...')
        pred_valid_mos = reg.predict(X_valid)

        SRCC = stats.spearmanr(pred_valid_mos, y_valid)
        logging.info("SRCC: {}".format(SRCC[0]))

        corr, _ = pearsonr(pred_valid_mos, y_valid)
        logging.info("PLCC: {}".format(corr))

    if args.do_test:
        logging.info('Loading pre-trained feature extractors...')
        Y_feature_extractor.load(args.model_dir)
        U_feature_extractor.load(args.model_dir)
        V_feature_extractor.load(args.model_dir)

        logging.info('Testing...')
        mos = []
        pred_mos_list = []
        for batch_images, batch_mos in load_data(args, batch_size=64):
            logging.info('Processing batch of size {}...'.format(len(batch_images)))
            mos.extend(batch_mos)
            
            train_Y_images_block_dct = jpeg_dct(batch_images, 'Y')
            train_U_images_block_dct = jpeg_dct(batch_images, 'U')
            train_V_images_block_dct = jpeg_dct(batch_images, 'V')

            y_features = Y_feature_extractor.transform(train_Y_images_block_dct)
            u_features = U_feature_extractor.transform(train_U_images_block_dct)
            v_features = V_feature_extractor.transform(train_V_images_block_dct)

            features = np.concatenate([y_features, u_features, v_features], axis=-1)
            
            if args.save:
                logging.info('Saving extracted features in {}'.format(args.output_dir))
                pickle.dump(features, open(os.path.join(args.output_dir, "features.pickle"), "wb"))

            reg = xgb.XGBRegressor(objective='reg:squarederror',
                                max_depth=5,
                                n_estimators=1500,
                                subsample=0.6,
                                eta=0.08,
                                colsample_bytree=0.4,
                                min_child_weight=4)

            reg.load_model(os.path.join(args.model_dir, 'xgboost.json'))

            logging.info('Testing...')

            pred_mos = []
            for start in range(0, features.shape[0], args.num_aug):
                test_features = features[start:start + args.num_aug]
                pred_test_mos = reg.predict(test_features)
                pred_mos.append(np.mean(pred_test_mos))

            pred_mos = np.array(pred_mos)
            SRCC = stats.spearmanr(pred_mos, mos)
            logging.info("SRCC: {}".format(SRCC[0]))

            corr, _ = pearsonr(pred_mos, mos)
            logging.info("PLCC: {}".format(corr))
            
            pred_mos_list.append(np.mean(pred_test_mos))
            
        pred_mos = np.array(pred_mos_list)
        mos = np.array(mos)
        
    if args.do_predict:
        logging.info('Loading pre-trained feature extractors...')
        Y_feature_extractor.load(args.model_dir)
        U_feature_extractor.load(args.model_dir)
        V_feature_extractor.load(args.model_dir)

        logging.info('Predicting...')
        for batch_images, _ in load_data(args, batch_size=64):
            logging.info('Processing batch of size {}...'.format(len(batch_images)))
            
            train_Y_images_block_dct = jpeg_dct(batch_images, 'Y')
            train_U_images_block_dct = jpeg_dct(batch_images, 'U')
            train_V_images_block_dct = jpeg_dct(batch_images, 'V')

            y_features = Y_feature_extractor.transform(train_Y_images_block_dct)
            u_features = U_feature_extractor.transform(train_U_images_block_dct)
            v_features = V_feature_extractor.transform(train_V_images_block_dct)

            features = np.concatenate([y_features, u_features, v_features], axis=-1)

            reg = xgb.XGBRegressor(objective='reg:squarederror',
                                max_depth=5,
                                n_estimators=1500,
                                subsample=0.6,
                                eta=0.08,
                                colsample_bytree=0.4,
                                min_child_weight=4)

            reg.load_model(os.path.join(args.model_dir, 'xgboost.json'))

            logging.info('Predicting...')

            pred_scores = []
            for start in range(0, features.shape[0], args.num_aug):
                test_features = features[start:start + args.num_aug]
                pred_test_scores = reg.predict(test_features)
                pred_scores.append(np.mean(pred_test_scores))

            pred_scores = np.array(pred_scores)
            logging.info("Predicted scores: {}".format(pred_scores))


if __name__ == '__main__':
    main(parse_args())
