const mongoClient = require('mongodb').MongoClient
const state = {
    db: null
}

module.exports.connect = function (callback) {
    //const url ='mongodb+srv://newUser1:newUser1@group06.ybqzk.mongodb.net/ITMS_website_db?retryWrites=true&w=majority'
    //const url ='mongodb://localhost:27017'
    const url = 'mongodb+srv://Merry-dbUser:Merry-dbUser@cluster0.hxvls.mongodb.net/ITMS-site?retryWrites=true&w=majority';
    const dbname = 'ITMS_website_db'

    mongoClient.connect(url, (err, data) => {
        if (err) return callback(err)
        state.db = data.db(dbname)
        // data.close();
        callback()
    })
}

module.exports.get = function () {
    // console.log(state.db)
    return state.db
}