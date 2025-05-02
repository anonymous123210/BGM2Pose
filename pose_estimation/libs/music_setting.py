from logging import getLogger
logger = getLogger(__name__)

def get_music_cv(ver:int = 1):
    if ver == 1:
        output = {'train':['arnor'], 'test':['arnor']}
    elif ver == 2:
        output = {'train':['cirrus', 'arnor', 'mantron'], 'test':['arnor']}
    elif ver == 3:
        output = {'train':['cirrus', 'arnor'], 'test':['mantron']}
    elif ver == 4:
        output = {'train':['arnor'], 'test':['mantron']}
    elif ver == 5:
        output = {'train':['cirrus','arnor','jazz'], 'test':['mantron']}
    elif ver == 6:
        output = {'train':['mantron'], 'test':['mantron']}
    elif ver == 7:
        output = {'train':['cirrus'], 'test':['cirrus']}
    elif ver == 8:
        output = {'train':['cirrus', 'mantron'], 'test':['arnor']}
    elif ver == 9:
        output = {'train':['mantron', 'arnor'], 'test':['cirrus']}
    elif ver == 10:
        output = {'train':['cirrus', 'arnor','jazz'], 'test':['mantron']}
    elif ver == 11:
        output = {'train':['cirrus', 'mantron','jazz'], 'test':['arnor']}
    elif ver == 12:
        output = {'train':['mantron', 'arnor','jazz'], 'test':['cirrus']}
    elif ver == 13:
        output = {'train':['mantron', 'arnor','cirrus'], 'test':['jazz']}
        
    else:
        message = f"dataset_name should be selected from 1,2,3."
        logger.error(message)
        raise ValueError(message)
    logger.info('train:')
    logger.info(output['train'])
    logger.info('test:')
    logger.info(output['test'])
    return output
    