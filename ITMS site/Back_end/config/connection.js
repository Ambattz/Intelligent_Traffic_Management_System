const { MongoClient } = require('mongodb');

class Mongo {
    constructor() {
        this.client = new MongoClient("mongodb://localhost:27017", {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
    }

    async main() {
        await this.client.connect();
        console.log('Connected to MongoDB');

        this.db = this.client.db();
    }
}

module.exports = new Mongo();