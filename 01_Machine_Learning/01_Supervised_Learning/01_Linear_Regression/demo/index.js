
// set the dimensions and margins of the graph
var margin = {top: 30, right: 30, bottom: 30, left: 60};
width = 1000 - margin.left - margin.right;
height = 800 - margin.top - margin.bottom;

randn = () => ((Math.random() * 2) - 1)

let learningRate = 0.1
const learningRateChange = 0.00005
// let coefficents = [randn(), randn(), randn(), randn(), randn()]
// let coefficents = [randn(), randn(), randn(), randn()]
let coefficents = [randn(), randn(), randn()]
// let coefficents = [randn(), randn()]
let xMin;
let xMax;
let yMin;
let yMax;


function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    let m = Math.floor(Math.random() * (i + 1));
    [array[i], array[m]] = [array[m], array[i]];
  }
}

function cost_der(x, target) {
  let actual = 0
  const coff = [...coefficents].reverse()
  coff.forEach((c, i) => {
    actual += c * (x ** i)
  })
  return actual - target
}

function step(points) {
  let m = points.length
  let X = points.reduce((a, b) => a + b[0], 0)
  let totalCost = points.reduce((acc, [x, target]) => {
    acc += cost_der(x, target)
    return acc
  }, 0)
  // dw, db
  return [totalCost*X/m, totalCost/m];
}

// this should work like, the delta should be much higher for higher order coefficents that drastically affect the cost
// function step(points, coffs) {
//   let m = points.length
//   let totalX = points.reduce((a, b) => a + b[0], 0)
//   let totalCost = points.reduce((acc, [x, target]) => {
//     acc += cost_der(x, target)
//     return acc
//   }, 0)
//   const dw = coffs.map((coff) => {
//     return (coff*((totalCost)/(m)))
//   })
//   return [dw, totalCost/m];
// }


function plotLine() {
  points = []
  const coff = [...coefficents].reverse()
  for (let x = xMin; x < xMax; x+=((xMax - xMin)/500)) {
    let y = 0
    coff.forEach((c, i) => {
      y += c * (x ** i)
    })
    points.push([x, y])
  }
  return points
}

document.addEventListener("DOMContentLoaded", () => {
  
  // data1 = data1.split('\n').map((tuple) => {
  //   return tuple.split(',').map(parseFloat)
  // })

  data1 = d3.range(300).map(function(){
    let variance = ((Math.random() * 2) - 0.5) * 2
    let x = Math.random() * 1.5
    let y = -3*(x**2) + 2*x + 10 + variance
    return [x, y]
  }),

  xMin = d3.min(data1, (d) => d[0])
  xMax = d3.max(data1, (d) => d[0])
  yMin = d3.min(data1, (d) => d[1])
  yMax = d3.max(data1, (d) => d[1])

  // append the svg object to the body of the page
  var svg = d3.select("#my_dataviz")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  // Add X axis
  var x = d3.scaleLinear()
    .domain([xMin, xMax])
    .range([ 0, width ]);
  svg.append('g')
    .attr('color', 'white')
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([yMin, yMax])
    .range([ height, 0]);
  svg.append("g")
    .attr('color', 'white')
    .call(d3.axisLeft(y));

  // Add dots
  svg.append('g')
    .selectAll('dot')
    .data(data1)
    .enter()
    .append('circle')
      .attr("cx", function (d) { return x(d[0]) } )
      .attr("cy", function (d) { return y(d[1]) } )
      .attr("r", 3)
      .style("fill", "#69b3a2")
  
  // console.log(plotLine([2, 1], xMin, xMax))
  let line = d3.line()
    .x(d => x(d[0]))
    .y(d => y(d[1]))
  
  let curve = svg.append('path')
    .attr('fill', 'none')
    .attr('stroke', 'cyan')
    .attr('stroke-width', 3)
    .attr('d', line(plotLine()))

    function SGD(in_data, batchSize) {
      shuffle(in_data)
      let lowerBatchSize = 0
      data = []
      for (let i = 0; i < in_data.length; i+=batchSize) {
        data.push(in_data.slice(lowerBatchSize, lowerBatchSize + batchSize))
        lowerBatchSize += batchSize
      }
       
      idx = 0
      const intervalId = setInterval(() => {
        const [dw, db] = step(data[idx])

        // for some reason negitive regularization helped achive a better polynomial fit
        // was running into a problem where the gradient would scal all coefficents equally
        // and they were constrained from escaping the reletive initial distances from each other
        // negitive reg rate allowed them to break free wtf??
        for (let i = 0; i < coefficents.length - 1; i++) {
          coefficents[i] = coefficents[i] - learningRate*dw
          // reg rate is related to the order of the coefficient
          // the higer the order the more it is allowed to roam free
          // coefficents[i] = (1-learningRate*(-1*(i)/300)) * coefficents[i] - learningRate*dw
          coefficents[i] = (1-learningRate*(-1*(coefficents.length - i)/300)) * coefficents[i] - learningRate*dw
          // coefficents[i] = (1-learningRate*((i)/300)) * coefficents[i] - learningRate*dw
          // coefficents[i] = (1-learningRate*((coefficents.length - i)/300)) * coefficents[i] - learningRate*dw
          // coefficents[i] -= learningRate*dw
        }

        coefficents[coefficents.length-1] -= learningRate*db

        // coefficents = coefficents.map((c, i) => {
        //   // return (c - (c > 0 ? 1 : -1)*learningRate*(10/500)) - ((learningRate/batchSize)  * (i >= coefficents.length-1 ? nabla : nabla * avgX))
        //   // return (1-learningRate*(10/500)) * c - ((learningRate/batchSize)  * (i >= coefficents.length-1 ? nabla : nabla * avgX))
        //   // return c - ((learningRate/batchSize)  * (i >= coefficents.length-1 ? nabla : nabla * avgX ))
        // })

        learningRate -= learningRateChange
        curve
        .transition().duration(100)
        .attr('d', line(plotLine()))

        idx++
        if (idx>=data.length) {
          clearInterval(intervalId)
          if (learningRate > 0) {
            SGD(data1, batchSize)
          } else {
            console.log(coefficents)
          }
        }
      }, 10)
    }

  SGD(data1, 3)
  // when ther is more varience in the data a smaller batchsixe creates a better fit

  // SOMETIMES it seems to help to use one degree more than you need and then throw away the biggest order coefficent tht goes to zero

})