import random
import requests
from bs4 import BeautifulSoup as bs4
import pandas as pd



class WS:
    def __init__(self, num_pages, link="http://www.ufcstats.com/statistics/events/completed?page="):
        self.df = pd.DataFrame()
        self.counter=1
        self.first_event = True
        for i in range(1,num_pages+1):
            self.scrape_page(link+str(1))
        self.save("training")


        

    def scrape_page(self, link):
        
        home_page = requests.get(link)
        events = [event.get('href') for event in bs4(home_page.content,'html.parser').findAll('a')]
        if self.first_event==True:
            events.pop(6)
            self.first_event=False
        for event in events:
            if event!=None and event.startswith("http://www.ufcstats.com/event-details/"):

                current_event = requests.get(event)
                temp = []
                
                fighter_list = [fighter.get('href') for fighter in bs4(current_event.content,'html.parser').find_all('a')]
                for fighter in fighter_list:                    
                    if fighter != None and fighter.startswith("http://www.ufcstats.com/fighter-details/"):
                        if self.counter % 2 != 0:
                            temp.append(self.get_fighter_stats(fighter))
                            self.counter += 1
                        else:
                            temp.append(self.get_fighter_stats(fighter))
                            self.append_col(temp)
                            temp.clear()
                            self.counter += 1
        

    def get_fighter_stats(self,fighter):
        stats_page = requests.get(fighter)
        stat_parser = bs4(stats_page.content, 'html.parser')
        content = []
        for liTag in stat_parser.find_all('li', {'class': 'b-list__box-list-item b-list__box-list-item_type_block'}):
            content.append(liTag.text.strip().replace("\n", "").replace("\t", " "))
            
        del content[0]
        del content[0]
        del content[1]
        del content[1]
        record = stat_parser.find('span', {'class':'b-content__title-record'}).text
        content.append(record.strip().replace("\t"," ").replace("\n",""))
        return content
    
    def append_col(self, arr):
        choice = random.choice([0,1])
        if choice==0:
            col = arr[0] + arr[1]
            col.pop(5)
            col.pop(15)
            col.append('W')
            self.df=pd.concat((self.df,pd.Series(col).rename(str(int(self.counter/2)))),axis=1)
         
        else:
            col = arr[1] + arr[0]
            col.pop(5)
            col.pop(15)
            col.append('L')
            self.df=pd.concat((self.df,pd.Series(col).rename(str(int(self.counter/2)))),axis=1)
    
    def save(self,file_name):
        file_name += ".csv"
        print(self.df)
        self.df.to_csv(file_name)

link = "http://www.ufcstats.com/statistics/events/completed?page="
WS(10,link)