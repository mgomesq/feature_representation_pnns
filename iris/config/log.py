import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 

logging.basicConfig(
    filename='log_iris.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

def log_this(str, level='info'):

    if level == 'info':
        logging.info(f'INFO: {str}')
    else: 
        print('Unknown level, logging as INFO')
        logging.info(f'INFO: {str}')