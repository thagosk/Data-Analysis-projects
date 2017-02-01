
# coding: utf-8

# # P3 - Data Wrangling with MongoDB
# 
# # Prepared by: Tedros Hagos

# OpenStreetMap (OSM) is a huge geographical information database, constantly updated by a large community of contributors all over the world. The idea of having constant input from developers always makes the data upto date, but at the same time, it is always prone to be not error free. In this project, I am going to explore for common errors and use the data wrangling techniques to fix some of the common errors programmatically.Initially our data is in XML format and then it is converted to Json data and then populated to MongoDB database.

# ### Choosen Area

# In this project, I used the open street map of Santa Cruz, a small town located in the state of California, USA.

# ### Overview of our dataset

# In[9]:

# import all the libraries needed to compile this project
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
import collections
import os


# In[12]:

# Define our osm file data name and path 
datadir = 'C:/Users/tedi/Desktop/tedros_data'
datafile = 'santa-cruz_california.osm'
cal_data = os.path.join(datadir, datafile)


# In[13]:

# Finds out how many unique users have contributed to 
# the map in this particular area.

def number_users(filename):  
    users = set() # initilize it to Null
    for _, element in ET.iterparse(filename):
        for e in element:   # loops through the elements
            if 'uid' in e.attrib:   # checks for unique users
                users.add(e.attrib['uid']) # adds unique user to the "users" dataset

    return users

users = number_users(cal_data)  
len(users)  


# As mentioned earlier, the open street map is open for users to develop, the above program code depicts the number of unique users who contributed for our OSM data.

# In[14]:

# Count the number of unique elements in our XML data to have a quick overview of our data structure
# The output is a dictionary with the tag name as the key and number of times this tag can 
# be encountered in the map as value
def count_tags(filename):  
        tags = {} # define an empity dictionary
        for event, elem in ET.iterparse(filename): # iterates throught each element in the XML file and  
            if elem.tag in tags:                   # add the relevant node name to a dictionary
                tags[elem.tag] += 1    
            else:                     #iterate through each node element and increment the dictionary 
                tags[elem.tag] = 1    #value for that node.tag key
        return tags
cal_tags = count_tags(cal_data)
pprint.pprint(cal_tags)


# In[15]:

lower = re.compile(r'^([a-z]|_)*$')  # matches lower case characters
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')  # matches lower case characters that have colon
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')  # matches characters that have a problem to be loaded to MongoDB

def key_type(element, keys):  
    if element.tag == "tag":
        for tag in element.iter('tag'):    # iteratest through the file to identify elements and assign to their 
            k = tag.get('k')               # respective categories in the dictionary
            if lower.search(k):      #checks kind of key and increments the respective category dictionary
                keys['lower'] += 1   
            elif lower_colon.search(k):
                keys['lower_colon'] += 1
            elif problemchars.search(k):
                keys['problemchars'] += 1
            else:
                keys['other'] += 1    # if keys are different from either of them, it classifies as "other"

    return keys

def process_map(filename):  
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0} # initialize the classification dictionary with null

    for _, element in ET.iterparse(filename):   # iterates through the file
        keys = key_type(element, keys)     # calls the above funtion: key_type

    return keys

keys = process_map(cal_data)  
pprint.pprint(keys)  


# ### Common Errors in our dataset

# #### Zip-Code

# In[16]:

# checks if Zip-code is in the correct format, 
# returns list of invalid zip codes

from collections import defaultdict

def audit_zipcode(invalid_zipcodes, zipcode):
    twoDigits = zipcode[0:2]
    
    if not twoDigits.isdigit():
        invalid_zipcodes[twoDigits].add(zipcode)
    
    elif twoDigits != 95:
        invalid_zipcodes[twoDigits].add(zipcode)
        
def is_zipcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_zip(datafile):
    osm_file = open(datafile, "r")
    invalid_zipcodes = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_zipcode(tag):
                    audit_zipcode(invalid_zipcodes,tag.attrib['v'])

    return invalid_zipcodes

san_zipcode = audit_zip(cal_data)


# #### Common zip code Errors

# In[17]:

pprint.pprint(dict(san_zipcode))


# In[18]:

# takes the invalid zipcodes and converts them to the expected zipcode format
# checks for prefix Letters, if they exist, it strips it away
# checks for extension digits after the 5 digit post code and trims them away if they exist
def update_name(zipcode):
    testNum = re.findall('[a-zA-Z]*', zipcode)
    if testNum:
        testNum = testNum[0]
    testNum.strip()
    if testNum == "CA":
        convertedZipcode = (re.findall(r'\d+', zipcode))
        if convertedZipcode:
            if len(convertedZipcode) == 2:
                return (re.findall(r'\d+', zipcode))[0] + "-" +(re.findall(r'\d+', zipcode))[1]
    if re.findall('(\d{5})-\d{4}', zipcode):  # if more than 5 digit
        return re.sub('(\d{5})-\d{4}', '\\1', zipcode) # removes the digits beyond the standard 5 digit numbers
    else:
        return (re.findall(r'\d+', zipcode))[0]

for zip, ways in san_zipcode.iteritems():  # iterates thrhough datafile
    for name in ways:
        better_name = update_name(name)  # update invalid zipcode, with expected once
        print name, "-->", better_name


# #### Street

# In[20]:

# create expected name list of Streets
from collections import defaultdict

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

# a list of non-abbreviated street types that we would expect them in our normal language 
expected = ["Avenue", "Boulevard", "Commons", "Court", "Drive", "Lane", "Parkway", "Seabright", "Chestnut", "Cedar",
             "front", "Place", "Road", "Square", "Street", "Pacific", "Front", "Circle", "Terrace", "Trail", "Way",
             "Ramps", "Merrill", "Esplanade", "Extension", "Gulch", "Loop", "Mar", "WAy", "9"]


# In[21]:

#function that uses a regular expression to check if 'street_name' is 
# in the 'expected' list of street types; if it is not, 'street_name' is
# added to the 'street_types' dict
def audit_street_type(street_types, street_name):  
    m = regex.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


# In[22]:

# function to check if the 'k' attribute of the xml tag identified as street address
def is_street_name(elem):  
    return (elem.attrib['k'] == "addr:street")


# In[23]:

# a function to generate a list of all those which we dont expect them (based on the list of expected)
# after going all through the 'nodes' and 'ways'

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_street_name(elem):    
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])

    return street_types


# In[24]:

san_street_types = audit(cal_data)


# In[25]:

pprint.pprint(dict(san_street_types))  # prints common errors in the name of streets


# In[26]:

# Mapping the Abrivated text to their full length
street_type_mapping = {'Ave'  : 'Avenue',  
                       'Blvd' : 'Boulevard',
                       'Dr'   : 'Drive',
                       'Ln'   : 'Lane',
                       'Pkwy' : 'Parkway',
                       'Rd'   : 'Road',
                       'WAy'  : 'Way',
                       'St'   : 'Street'}


# In[27]:

# function to update the street names based on regex and 'mapping' dictionary which has was created above cell.
def update_street_name(name, mapping, regex):  
    m = regex.search(name)
    if m:
        street_type = m.group()
        if street_type in mapping:
            name = re.sub(regex, mapping[street_type], name)

    return name


# In[28]:

# updating the street names in the expected format
for street_type, ways in san_street_types.iteritems():  
    for name in ways:
        updated_street_name = update_street_name(name, street_type_mapping, street_type_re)
        print name, "-->", updated_street_name


# In[19]:

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint
import codecs
import json

OSMFILE = "santa-cruz_california.osm"
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

#pre-compiled regex queries
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def shape_element(e):
    node = {}
    node['created'] = {}
    node['pos'] = [0,0]
    if e.tag == "way":
        node['node_refs'] = []
    if e.tag == "node" or e.tag == "way" :
        node['type'] = e.tag
        #attributes
        for k, v in e.attrib.iteritems():
            #latitude
            if k == 'lat':
                try:
                    lat = float(v)
                    node['pos'][0] = lat
                except ValueError:
                    pass
            #longitude
            elif k == 'lon':
                try:
                    lon = float(v)
                    node['pos'][1] = lon
                except ValueError:
                    pass
            #creation metadata
            elif k in CREATED:
                node['created'][k] = v
            else:
                node[k] = v
        #children
        for tag in e.iter('tag'):
            k = tag.attrib['k']
            v = tag.attrib['v']
            
            if problemchars.match(k):
                continue
            if lower.search(k):
                node['k']= v

            elif lower_colon.search(k):
                k_split = k.split(':')
                #address fields
                if k_split[0] == 'addr':
                    k_item = k_split[1]
                    if 'address' not in node:
                        node['address'] = {}
                    #streets
                    if k_item == 'street':
                       v = update_street_name(v, street_type_mapping, street_type_re)                 
                    #postal codes
                    elif k_item == 'postcode':
                        v = update_name(v)
                    node['address'][k_item] = v
                    continue   
            else:
              
                node['k']=v 
        return node
        #way children
    if e.tag == "way":
            for n in e.iter('nd'):
                ref = n.attrib['ref']
                node['node_refs'].append(ref);
            return node
    else:
        return None

def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data


process_map(cal_data)


# ## Data Overview with MongoDB

# We use Mongodb to visualise our data in aggregated format. Since, Mongodb can not read XML data format, we have to transfer our XML data in to json type format and populate it in to the mongodb database.

# In[20]:

data_source = process_map(cal_data)


# In[21]:

from pymongo import MongoClient


# In[22]:

client  = MongoClient('mongodb://localhost:27017')
db = client.examples


# In[23]:

[db.santa2_cruz_osm.insert(e) for e in data_source]


# #### OSM, JSON file Size in MB

# In[26]:

import os
print 'The original OSM file is {} MB'.format(os.path.getsize(cal_data)/1.0e6) # convert from bytes to megabytes
print 'The JSON file is {} MB'.format(os.path.getsize(cal_data + ".json")/1.0e6) # convert from bytes to megabytes


# #### Number of Unique users

# In[27]:

len(db.santa2_cruz_osm.distinct('created.user'))


# ####   # of Documents

# In[25]:

db.santa1_cruz_osm.find().count()


# #### Number of Nodes and Ways

# In[36]:

print "Number of nodes:",db.Santa_osm.find({'type':'node'}).count()
print "Number of ways:",db.Santa_osm.find({'type':'way'}).count()


# #### Top 10 Amenities (Public Service Areas)

# In[37]:

amenity = db.Santa_osm.aggregate([{'$match': {'amenity': {'$exists': 1}}},                                 {'$group': {'_id': '$amenity',                                             'count': {'$sum': 1}}},                                 {'$sort': {'count': -1}},                                 {'$limit': 10}])
print(list(amenity))


# #### Top 10 Restaurants

# In[39]:

cuisine = db.Santa_osm.aggregate([{"$match":{"amenity":{"$exists":1},
                                 "amenity":"restaurant",}},      
                      {"$group":{"_id":{"Food":"$cuisine"},
                                 "count":{"$sum":1}}},
                      {"$project":{"_id":0,
                                  "Food":"$_id.Food",
                                  "Count":"$count"}},
                      {"$sort":{"Count":-1}}, 
                      {"$limit":10}])
print(list(cuisine))


# #### Building Types

# In[38]:

building = db.Santa_osm.aggregate([
       {'$match': {'building': { '$exists': 1}}}, 
        {'$group': {'_id': '$building',
                    'count': {'$sum': 1}}}, 
        {'$sort': {'count': -1}},
        {'$limit': 5}])
print(list(building))


# #### Top 10 Freqent Post codes

# In[24]:

postcode = db.santa2_cruz_osm.aggregate( [ 
    { "$match" : { "address.postcode" : { "$exists" : 1} } }, 
    { "$group" : { "_id" : "$address.postcode", "count" : { "$sum" : 1} } },  
    { "$sort" : { "count" : -1}},
      {"$limit":10}] )
print(list(postcode))


# #### City Names

# In[42]:

city = db.Santa_osm.aggregate( [ 
    { "$match" : { "address.city" : { "$exists" : 1} } }, 
    { "$group" : { "_id" : "$address.city", "count" : { "$sum" : 1} } },  
    { "$sort" : { "count" : -1}},
      {"$limit":10}] )
print(list(city))

