var http = require('http')
const express = require('express')
const path = require('path')
const axios = require('axios')
const cors = require('cors')
require('dotenv').config()

const app  = express()
app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(express.static(path.join(__dirname, '../client/build')))
app.use(cors({
    origin: '*'
}))

app.post('/make-prediction', (req, res) => {
    const url = `${process.env.SERVER_URL}/score`
    const body = req.body
    const config = {
        headers: { 
          Authorization: `Bearer ${process.env.API_KEY}`,
          ContentType: 'application/json'
        }
    }
    axios.post(url, body, config).then(result => {
        res.send(result.data)
    }).catch(error => {
        console.error(error)
    })
})

const server = http.createServer(app)
server.listen(5000)
console.log('App running on port 5000')