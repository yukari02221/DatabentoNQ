from logging_utils import CustomLogger



def main():
    main_logger = CustomLogger("Main")
    main_logger.log('INFO', 'Main', "log started")

if __name__=="__main__":
    main()