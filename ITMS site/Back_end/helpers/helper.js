var express = require('express');
var db = require('../Mongo/mongo')
var collection = require('../config/collections')

module.exports = {
    print: (details, callback) => {
        console.log(details)
        console.log(db)
        details.password = bcrypt.hash(details.password, 10)
        db.get().collection('PoliceAccount').insertOne(details).then((client) => {
            callback(true)
        })
    },
    getAllDetails: () => {
        return new Promise(async (resolve, reject) => {
            let det = await db.get().collection(collection.LANE_COLLECTION).find().toArray()
            // console.log(det[0].Lane)
            console.log(det)
            resolve(det)
        })
    },
    findVehicle: (vehicle_no, callback) => {
        console.log(vehicle_no)
        console.log(db)
        db.get().collection('VehiclePlate').insertOne(vehicle_no).then((client) => {
            callback(true)
        })

    }
}
