import itchat
import re  
from itchat.content import TEXT,MAP,NOTE,SHARING,CARD,PICTURE,VOICE

dialogs = {}

'''
Auto reply specific message which contains certain string. Since every person usually send
their wishes only once a time, so there is no need to keep a dictionary that stores 'UserName 
and NickName' pair.
'''
# @itchat.msg_register([TEXT])
# def text_reply(msg):
#     match=re.search('新年',msg['Text'])
#     if match:
#         sender=''
#         friends = itchat.get_friends()[1:]
#         for friend in friends:
#             if friend['UserName']==msg['FromUserName']:
#                 if friend['RemarkName']:
#                     sender=friend['RemarkName']
#                 else:
#                     sender=friend['NickName']
#                 itchat.send(('谢谢！也祝你新年快乐哈'+sender),msg['FromUserName'])
#                 break

'''
Auto reply all kinds of messages.
'''
@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING, PICTURE,VOICE])
def text_reply(msg):
    sender = ''
    if msg['FromUserName'] in dialogs.keys():
        sender=dialogs.get(msg['FromUserName'])
    else:
        real_sender=get_Name(msg['FromUserName'])
        dialogs.update({msg['FromUserName']:real_sender})
        sender=real_sender
    
    itchat.send('【自动回复】%s你好，你的%s类型消息我已经收到。我现在有事不在，稍后联系你【自动回复】'%(sender,msg['Type']), msg['FromUserName'])


'''
return the name according to key UserName. If the users has a RemarkName then the method
will return the RemarkName preferentially.
'''
def get_Name(UserName):
    friends = itchat.get_friends()
    name=''
    for friend in friends:
        if friend['UserName']==UserName:
            if friend['RemarkName']:
                name=friend['RemarkName']
            else:
                name=friend['NickName']
    return name

itchat.auto_login(enableCmdQR=2)  
itchat.run()