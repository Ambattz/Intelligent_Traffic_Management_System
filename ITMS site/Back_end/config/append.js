'use strict';
var mongodb = require('mongodb');
var url = 'mongodb+srv://Merry-dbUser:Merry-dbUser@cluster0.hxvls.mongodb.net/ITMS-site?retryWrites=true&w=majority';
var MongoClient = mongodb.MongoClient;
var fs = require('fs');
var csv = require('fast-csv');
var dataArray = [];
var finishedReading = false;
//set interval limit as you like 
var interval = 1000 * 60;
var collection
var db

MongoClient.connect(url, function(err, db) {
    if (err) {
        console.log('Unable to connect to the mongoDB server. Error:', err);
    } else {
        db = db.db('ITMS_website_db');
        console.log('Connection established to', url);
        collection = db.collection('Lane');
    }
})

var stream = fs.createReadStream('data.csv');
var csvStream = csv
    .parseStream(stream, { headers: true })
    .on("data", function(data) {
        console.log("Start of parsing...");
        dataArray.push(data)
    })
    .on("end", function(data) {
        finishedReading = true;
        console.log("End of parsing");
        console.log("dataArray SUCCESS")
    });

var intervalFn = setInterval(function() {
    if (dataArray.length >= 1) {
        var twoItems = dataArray.splice(0, 1);
        console.log(new Date() + twoItems);
        collection.insert({ 'data': twoItems[0] });
    } else if (finishedReading) {
        //clean up if items left in array are less than 10 and also clear this interval function
        console.log(new Date() + dataArray.length);
        clearInterval(intervalFn);
    }
}, interval)