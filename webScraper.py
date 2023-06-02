import random
import requests
from bs4 import BeautifulSoup as bs4
import pandas as pd





class Scraper:

    def __init__(self, home_url, num_pages):
        self.df = pd.DataFrame()
        
        for i in range(1,num_pages+1):
            url = home_url+str(i)    
            print(url)
            self.main(url)
        self.save("new")

    def main(self,home_url):
        home_page = requests.get(home_url)
        parser = bs4(home_page.content, "html.parser")
        flag =1
        for event in parser.find_all('a'):
            e = event.get('href')
            
            if e != None and e[0:38] == "http://www.ufcstats.com/event-details/"and flag!=1:
                fight = requests.get(e)
                event_parser = bs4(fight.content, 'html.parser')
                counter = 1
                temp = []
                for fighter in event_parser.findAll('a'):
                    #extract names 
                    fighter = fighter.get('href')
                    if fighter != None and len(fighter) > 40 and str(fighter[0:40]) == "http://www.ufcstats.com/fighter-details/":
                        
                        if counter % 2 != 0:
                            temp.append(self.get_fighter_stats(fighter))
                            counter += 1
                        else:
                            temp.append(self.get_fighter_stats(fighter))
                            self.append_col(temp,counter)
                            temp.clear()
                            counter += 1
            flag+=1
        print("Event Completed")

    def save(self,file_name):
        file_name += ".csv"
        print(self.df)
        self.df.to_csv(file_name)
            

                 


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
    #use fight name as column name 
    #don't forget to append outcome 
    #
    
    def append_col(self, arr,counter):
        choice = random.choice([0,1])
        if choice==0:
            col = arr[0] + arr[1]
            col.pop(5)
            col.pop(15)
            col.append('W')
            self.df[counter/2]=col  
         
        else:
            col = arr[1] + arr[0]
            col.pop(5)
            col.pop(15)
            col.append('L')
            self.df[counter/2]=col
            


            

#    pass
#arr = ['        Reach:            69', '            SLpM:                    7.24', '            Str. Acc.:                    47%', 'SApM:                    4.89', '            Str. Def:                    59%', '', '            TD Avg.:                    0.29', '        TD Acc.:                    53%', '            TD Def.:                    84%', '            Sub. Avg.:                    0.3', '      Record: 23-7-0            ']
#print(arr)
home_url = "http://www.ufcstats.com/statistics/events/completed?page="
scraper = Scraper(home_url,2)

#change init to main massive increment page
    

