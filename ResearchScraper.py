# Tyler Crabtree
# Web Scraper (general)

#crucial imports (please install if you want program to function)
import wikipedia
import warnings
from unidecode import unidecode


#This grabs information off a typical wiki page
def ultraGeneric(topic, target):
    warnings.filterwarnings('ignore')
    topic = wikipedia.page(topic)                  # get page
    print("Title :" + topic.title)                 # print title
    target.write("Title :" + topic.title+"\n")
    print("Summary :" + topic.summary)             # print summary
    
    sum =("Summary :" + topic.summary+"\n").encode('utf-8')
    target.write(sum)
    print("Image: " + topic.images[0])             # print image
    target.write("Image: " + topic.images[0]+"\n")

    print("Categories: " + topic.categories[1]+ \
    ", "  + topic.categories[0] + ", " \
    + topic.categories[2])                         # print categories
    target.write("Categories: " + topic.categories[1]+ \
    ", "  + topic.categories[0] + ", " \
    + topic.categories[2]+"\n")

    title = topic.title                            #save variable names, get title
    summary = topic.summary                        #get summary
    image = topic.images[0]                        #get image
    cat = topic.categories[0]                      #to condense, only chose one category  (simplify variables if you wanted to throw this in an database)
    reception = receptionParse(topic)              # call function that parses for the critical reception
    reception = reception.lstrip()                 #just in case gets rid of extra whitespace
    print ("Reception:" + reception)               #print
    target.write("Reception:" + reception+"\n")
    target.write("\n")


def clean(word):                                    #makes sure strings are readable
    weirdError = ""
    word = word.replace( "'",weirdError)            # fixed error, parsing became confused by " ' " symbols, so i just took them out
    word = unidecode(word)                          # glorious, get rid of Japanese letters import
    return word

def shorten(reception):
    if(len(reception) > 200):                       # if too long
        clip = len(reception)/2
        reception = reception[:-(clip)]             # clip the length
        reception = reception + "..."               # add this for practical purposes
        return  reception



#the next three functions trim the length, I should have passed a second value in and limited by that size, but this works perfectly fine
#a little verbose though
def trim(summary):
        while(len(summary) > 40):   #shorten length of string function5
            summary = summary[:-1]
        summary = summary + "..."
        return  summary

def mediumTrim(summary):
    while (len(summary) > 190):  # shorten length of string function
        summary = summary[:-1]

    summary = summary + "..."
    return summary


#looks at the reception section, and grabs it
def receptionParse(r):
    receptionString = ""                                                # set empty screen
    i = 0;                                                              #build iterator
    while (i < len(
            r.content) - 12):                                     # '= Reception =" is in the middle of the conent section, so I end a -12 early to avoid overflow (shouldn't change data though)
                                                                        # the line below parses the content of a page and searches for a specific word
        s1 = (
        r.content[i] + r.content[i + 1] + r.content[i + 2] + r.content[i + 3] + r.content[
            i + 4] + r.content[i + 5] + r.content[i + 6] + r.content[i + 7] + r.content[i + 8] +
        r.content[i + 9] + r.content[i + 10] + r.content[i + 11] + r.content[i + 12])
        s2 = '= Reception ='
        if (s1 == s2):                                                  # checks to see if specific word was found
            i = i + 14;                                                 # offest to start section
            count = 0;

            while (i < len(r.content) - 12):                      # avoid overflow + set up to print section
                if (r.content[i] == '='):                         # break when next section is hit
                    #sys.stdout.write("Reception:" + (receptionString))  # print section
                    break  # exit to main menu
                receptionString = receptionString + r.content[i]  # append string
                if(r.content[i]  == '='):
                    count = count +1                                     # I don't want the reception to be too long, so stop at two sentances
                    if (count%1 == 0):
                        break
                i = i + 1  # iterate
        i = i + 1  # iterate
    return receptionString

# welcome and option prompt below, nothing too fancy
if __name__ == "__main__":
    print("Research Project Web Scraper'.")
    warnings.filterwarnings('ignore')
    target = open('Research.txt', 'w')

    while(1):
         check = raw_input("Look up topics to reasearch: ")
         if(check == 'exit'):
                break
         check = unidecode(check)
         ultraGeneric(check, target)
         print "type 'exit' to end reasearch"

