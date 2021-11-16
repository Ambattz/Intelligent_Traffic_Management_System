var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const mongo = require('./Mongo/mongo')
var session = require('express-session')


var indexRouter = require('./routes/index')
var usersRouter = require('./routes/users');

var app = express();

mongo.connect((err) => {
    if (err) console.log('Connection Error' + err);
    else console.log('Database Connected!')
})

var livereload = require('livereload')
const connectLivereload = require("connect-livereload");
liveReloadServer = livereload.createServer();
//liveReloadServer.watch(path.join(__dirname, 'public'));
liveReloadServer.server.on("connection", () => {
    setTimeout(() => {
        liveReloadServer.refresh("http://localhost:3000/live_traffic/live_traffic");
    }, 45000);
});


app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.set('view engine', 'hbs');
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
app.use(session({ secret: "Key" }))

app.use('/', indexRouter);
app.use('/users', usersRouter);

app.use(connectLivereload());

app.use(function(req, res, next) {
    next(createError(404));
});
// error handler
app.use(function(err, req, res, next) {
    // set locals, only providing error in development
    res.locals.message = err.message;
    res.locals.error = req.app.get('env') === 'development' ? err : {};
    // render the error page
    res.status(err.status || 500);
    res.render('error');
});

module.exports = app;