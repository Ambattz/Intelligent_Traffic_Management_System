var express = require('express');
var router = express.Router();
var path = require('path');
var db = require('../Mongo/mongo');
var helper = require('../helpers/helper')
var user_login_helper = require('../helpers/user_login_helper');
var changeStreams = require('../helpers/changeStreams');
var LocalStorage = require('node-localstorage').LocalStorage;



localStorage = new LocalStorage('./scratch');
//var store = require('store')
//var store = require('storejs')
//import {store} from 'storejs';
// store('test', 'tank', 1)
{/* <script type="text/javascript">
  store('test', 'tank');
</script> */}

localStorage.setItem("tech", "JavaScript");
/* GET home page. */

router.get('/', function(req, res, next) {
 // res.render(__dirname+'/Front_end/live/live.html');
  res.sendFile(__dirname + '/Front_end/Home/home.html');
  
  next();
});

router.get('/login', function(req, res, next) {
   res.sendFile(__dirname + '/Front_end/main/index.html');
   
   next();
});
  c1 = 0
  c2 = 0
  c3 = 0
  c4 = 0
router.get('/live_traffic/live_traffic', function(req, res, next) {
  //res.sendFile(__dirname + '/Front_end/live/live.html')
  //res.render('waiting',{})
  
  console.log("Current count in 4 lanes: "+c1+","+c2+","+c3+","+c4)
  console.log('Data Received')
  if(c1>0 && c2>0 && c3>0 && c4>0){
    res.render('live_traffic',{c1,c2,c3,c4})
  }
  else{
    res.render('waiting',{})
  }
  lane_no = changeStreams.result.data.Lane
  console.log('Updated Lane is: '+lane_no);
 
  
  if (lane_no == '1'){
    localStorage.setItem('data_01', changeStreams.result.data.Count);
   // console.log(localStorage)
     c1=localStorage.getItem('data_01')
     console.log('c1='+c1)
  }
  else if(lane_no=='2'){
    localStorage.setItem('data_02', changeStreams.result.data.Count);
    c2=localStorage.getItem('data_02')
    console.log('c2='+c2)
  }
  else if(lane_no=='3'){  
    localStorage.setItem('data_03', changeStreams.result.data.Count);
    c3=localStorage.getItem('data_03')
    console.log('c3='+c3)
    
  }
  else if(lane_no=='4'){
    localStorage.setItem('data_04', changeStreams.result.data.Count);
    c4=localStorage.getItem('data_04')
    console.log('c4='+c4)
    
  }

  
  next();
  

})
router.get('/statistics/statistics', function(req, res, next) {
  res.render('statistics',{})
  next();
});

router.get('/catch_vehicle/catch_vehicle', function(req, res, next) {
  res.render('catch_vehicle',{style:'catch_vehicle.css'})
  next();
});

router.get('/data_report/data_report', function(req,res,next){
  helper.getAllDetails().then((data)=>{
    console.log('Data Received')
    res.render('index',{data})
    next();
  })
})
//POST
router.post('/login', function(req, res, next) {
  user_login_helper.doLogin(req.body).then((response)=>{
    console.log(response.status);
    if(response.status){
      req.session.loggedIn = true
      req.session.user = response.user
      res.redirect('/login')
    }else{
      res.redirect('/')
    }
    next();
  })  

});
router.post('/submit', function(req, res, next) {
    user_login_helper.doSignup(req.body).then((response)=>{
      console.log(response);
    })
    res.redirect('/');
    next();
  })
router.post('/findVehicle', function(req, res, next) {
    helper.findVehicle(req.body,(result)=>{
      console.log("Data entered")
      next();
    })
  })

module.exports = router;

