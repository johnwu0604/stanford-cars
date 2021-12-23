import './App.css'
import React from 'react'
import Button from '@mui/material/Button'
import TextField from '@mui/material/TextField'
import Grid from '@mui/material/Grid'
import axios from 'axios'

class App extends React.Component {

  constructor(props) {
    super(props)
    this.state = { 
      image_input: 'https://file.kelleybluebookimages.com/kbb/base/house/1997/1997-Dodge-Grand%20Caravan%20Passenger-FrontSide_DTGCVES961_506x375.jpg',
      image_url: '',
      prediction: ''
    }
    this.handleChange = this.handleChange.bind(this)
    this.handleSubmit = this.handleSubmit.bind(this)
  }

  handleChange(event) {
    this.setState({image_input: event.target.value})
  }

  async handleSubmit(event) {
    event.preventDefault()
    const url = '/make-prediction'
    const body = { "image_url": this.state.image_input }
    try {
      const result = await axios.post(url, body)
      this.setState({
        image_url: this.state.image_input,
        prediction: result.data.prediction
      })
    } catch (error) {
      console.error(error)
      return false
    }
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <p>
            Input the links to a photo of one of the <a href="https://github.com/johnwu0604/stanford-cars/blob/main/src/classes.txt" target="_blank">following car brands</a> and click submit to run prediction.
          </p>
          <form onSubmit={this.handleSubmit}>
            <Grid container spacing={2} direction="row">
              <Grid item xs={8}>
                <TextField size="medium" fullWidth="true" value={this.state.image_input} onChange={this.handleChange}/>
              </Grid>
              <Grid item xs={4}>
                <Button variant="contained" type="submit">Submit</Button>
              </Grid>
            </Grid>
          </form>
          {this.state.prediction && <h5>Prediction: {this.state.prediction}</h5>}
          <img
            src={this.state.image_url}
            loading="lazy"
          />
          <br/>
          <a
            className="App-link"
            href="https://github.com/johnwu0604/stanford-cars"
            target="_blank"
            rel="noopener noreferrer"
          >
            Project Github Repo
          </a>
        </header>
      </div>
    )
  }
}

export default App
