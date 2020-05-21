
import time
import datetime
import re
import pandas as pd
from string import printable
import numpy as np



def remove_clean(f):
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)
    f = f.translate(translator)
    f=f.strip()
    if len(f)==0:
        f=''
    return f

def cleanMessage(message):
    # Remove new lines within message
    cleanedMessage = message.replace('\n',' ~ ').lower()
    # Deal with some weird tokens
    cleanedMessage = cleanedMessage.replace("\xc2\xa0=", "")
    # Remove punctuation
    cleanedMessage = re.sub('([\'.,!?])','', cleanedMessage)
    # Remove multiple spaces in message
    #cleanedMessage = re.sub(' +',' ', cleanedMessage)
    
    cleanedMessage=remove_clean(cleanedMessage)
    return cleanedMessage

def getimestamp(s):
    try:
        
        return time.mktime(datetime.datetime.strptime(s, "%m/%d/%y, %I:%M %p").timetuple())
    except:
        print(s)


def getdif(s1,s2):
    return getimestamp(s2)-getimestamp(s1)


def read_from_user(filename,namesen,yourname):

	file=open(filename,encoding='latin1')
	content=file.read()
	file.close()

	content=content.splitlines()
	dialogs=[]


	
	partner=namesen

	for line in content:
	    if 'Messages to this chat and calls are now secured with end-to-end encryption. Tap for more info.' in line:
	        continue
	    elif '<Media omitted>' in line:
	        continue
	    elif 'Missed voice call' in line:
	        continue
	    elif 'Missed video call' in line:
	        continue
	        
	    else:
	        text = re.sub(r'http\S+', '', line)
	        if len(text)>0 and len(text)<350:
	            dialogs.append(text)
	           
	date_wise=[]

	for line in dialogs:
	    try:
	        parts=line.split(' - ')
	        if len(parts)>2:
	            continue
	        timestamp=parts[0]
	        usersplit=parts[1]
	        message=usersplit.split(':', 1)[1]
	        sender=usersplit.split(':', 1)[0]
	        if partner in sender.lower():
	            sender=partner
	        elif yourname in sender.lower():
	            sender=yourname
	        date_wise.append([timestamp.strip(),sender,message.strip()])
	    except:
	        continue
	        
	        



	join_lines=[date_wise[0]]


	for line in range(1,len(date_wise)):
		try:
		    cur_text=date_wise[line]
		    last=join_lines[-1]
		    #print(last)
		    last_ts=getimestamp(last[0])
		    current_ts=getimestamp(cur_text[0])
		    if current_ts-last_ts<1000:
		        last_sender=last[1]
		        current_sender=cur_text[1]
		        if last_sender==current_sender:
		            last_mesg=last[2]
		            cur_mesg=cur_text[2]
		            last_mesg+='\n'+cur_mesg
		            join_lines.pop()
		            join_lines.append([last[0],last_sender,last_mesg])
		        else:
		            join_lines.append(cur_text)
		            
		    else:
		        join_lines.append(cur_text)
		except:
			print(line)

	convos=[[],[]]

	for i in range(len(join_lines)):
	    cur_mes=join_lines[i]
	    cur_user=cur_mes[1]
	    cur_text=cur_mes[2]
	    cur_ts=cur_mes[0]
	    try:
	        next_mes=join_lines[i+1]
	        next_user=next_mes[1]
	        next_text=next_mes[2]
	        next_ts=next_mes[0]
	        if next_user==cur_user:
	            continue
	        if cur_user==partner:
	            
	            if getdif(next_ts,cur_ts)<1000:
	                convos[0].append(cur_text)
	                convos[1].append(next_text)
	                #convos.append([cur_text,next_text])
	        else:
	            
	            continue
	            
	        
	    except IndexError:
	        continue
	    
	    

	cleansed_convos=[]

	for i in convos:
	    x=[]
	    for message in i:
	        message=cleanMessage(message)
	        x.append(message)
	    cleansed_convos.append(x)
	        


	df=pd.DataFrame(cleansed_convos)
	df=df.transpose()
	df.columns=[partner,yourname]


	df=df.applymap(lambda y: ''.join(filter(lambda x: x in printable, y)))

	last_processing=df.values.tolist()

	final_list=[]


	for i in range(len(last_processing)):
	    try:
	        exchange=last_processing[i]
	        if len(exchange[0])==0 and len(exchange[1])==0:
	            #(1)
	            continue
	        elif len(exchange[0])==0 and len(exchange[1])>0:
	            #print(2)
	            final_list[-1][1]+=' ~ '+ exchange[1]
	        elif len(exchange[0])>1 and len(exchange[1])==0:
	            #print(3)
	            final_list[-1][0]+=' ~ '+ exchange[0]
	            
	        else:
	            final_list.append(exchange)
	            
	            
	                
	    except IndexError:
	        print('what?')
	
	return final_list

filenames=['WhatsApp Chat with Samarth Gujju Boi.txt','WhatsApp Chat with Aryan.txt','WhatsApp Chat with Aditi.txt','WhatsApp Chat with Anmol.txt','WhatsApp Chat with Aditya.txt','WhatsApp Chat with Emarti Sen.txt','WhatsApp Chat with Mehul Panda.txt','WhatsApp Chat with Ishika VTC.txt','WhatsApp Chat with manasa robovitics.txt']
names=['samarth','aryan','aditi','anmol','aditya','emarti','mehul','ishika','manasa']
yourname='shrawan'

convos=[]
for i in range(len(filenames)):
	print(filenames[i])
	d=read_from_user(filenames[i],names[i],yourname)
	print(d)
	convos=convos+d

df=pd.DataFrame(convos)
df.columns=['partner',yourname]
#print(df)

final_dict={}
for i in convos:
    final_dict.update({i[0]:i[1]})


np.save('conversationDictionary.npy', final_dict)

conversationFile = open('conversationData.txt', 'w')
for key, value in final_dict.items():
    if (not key.strip() or not value.strip()):
        # If there are empty strings
        continue
    conversationFile.write(key.strip() + value.strip())

print(df)


