const { MongoClient } = require('mongodb');
var collections = require('../config/collections')

function closeChangeStream(timeInMs = 60000, changeStream) {
    return new Promise((resolve) => {
        setTimeout(() => {
            console.log("CLOSING CHANGE STREAM.........");
            changeStream.close();
            resolve();
        }, timeInMs)
    })
};
async function main() {
    const uri = 'mongodb+srv://Merry-dbUser:Merry-dbUser@cluster0.hxvls.mongodb.net/ITMS-site?retryWrites=true&w=majority';
    const client = new MongoClient(uri, { useUnifiedTopology: true });
    try {
        // Connect to the MongoDB cluster
        await client.connect();
        // Make the appropriate DB calls
        const pipeline = [
            {
                '$match': {
                    'operationType': 'insert'

                },
            }
        ];
        //await monitorListingsUsingEventEmitter(client);
        await monitorListingsUsingEventEmitter(client, 600000, pipeline);
    } finally {
        // Close the connection to the MongoDB cluster
        await client.close();
    }
}

main().catch(console.error);

async function monitorListingsUsingEventEmitter(client, timeInMs = 60000, pipeline = []) {
    console.log("STARTED STREAMING...........")
    const collection = client.db("ITMS_website_db").collection(collections.LANE_COLLECTION)
    const changeStream = collection.watch(pipeline);
    changeStream.on('change', (next) => {
        console.log(next.fullDocument)
        module.exports.result = next.fullDocument

    });
    await closeChangeStream(timeInMs, changeStream);
}
