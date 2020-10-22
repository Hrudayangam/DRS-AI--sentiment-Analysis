# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:59:33 2020

@author: hruda
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 02:54:42 2020

@author: Asus

from Text_Sentiment import *



okay = "bunty is bad negative ew"



"""




#Creating GUI with tkinter


from Text_Sentiment_ import predict_review


import tkinter
from tkinter import *






def chatbot_response(msg):
    res =  predict_review(msg)
    
    return res




def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "DRS-AI: " + "Please enter what you want to analyse:" + '\n\n')
        
        
        ChatLog.insert(END, "You: " + msg+ '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "DRS-AI: " + res + '\n\n')
        
        
        ChatLog.insert(END, "DRS-AI: " + "Please enter what you want to analyse:" + '\n\n')
        
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        
        
 

base = Tk()
base.title("DRS-AI")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="cyan", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()



