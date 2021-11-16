var db = require('../Mongo/mongo')
var collection = require('../config/collections')
const bcrypt = require('bcrypt')

module.exports = {
    doSignup: (userData) => {
        return new Promise(async(resolve, reject) => {
            userData.password = await bcrypt.hash(userData.password, 10)
            db.get().collection('user').insertOne(userData).then((data) => {
                resolve(data.ops[0])
            })
        })
    },
    doLogin: (userData) => {
        return new Promise(async(resolve, reject) => {
            let loginStatus = false
            let response = {}
            console.log(userData)
            let user = await db.get().collection(collection.USER_COLLECTION).findOne({ emailAdress: userData.loginemail })
            if (user) {
                bcrypt.compare(userData.loginPassword, user.password).then((status) => {
                    if (status) {
                        console.log("Login Success")
                        response.user = user
                        response.status = true
                        resolve(response)
                    } else {
                        console.log("Login Failed")
                        resolve({ status: false })
                    }
                })
            } else {
                console.log('Login Failed:Email')
                resolve({ status: false })
            }
        })
    }
}