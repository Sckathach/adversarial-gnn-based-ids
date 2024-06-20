#import "@preview/ctheorems:1.1.2": *
#import "@preview/showybox:2.0.1": showybox

#let rainbow(content, color_from: orange, color_to: purple, angle: 70deg) = {
  set text(fill: gradient.linear(color_from, color_to, angle: angle))
  box(content)
}

#let celeste(x, padding: 0.7, color_from: orange, color_to: purple, angle: 70deg, amplitude: -0.5, freq: 0.4, start: 0 * calc.pi) = {
  x = x.split("").slice(1, -1)
  for i in range(x.len()) {
    if x.at(i) == " " {
      x.at(i) = h(0.3em)
    }
    else {
      x.at(i) = [#{text(x.at(i), fill: white) + place(x.at(i), dy: amplitude*calc.sin(freq*i + start)*1em - padding*1em)}]
    }
  }
  rainbow(
    grid(columns: x.len(), ..x),
    color_from: color_from,
    color_to: color_to,
    angle: angle
  )
}

#let colors = (
  "blue": rgb("#2196F3"),
  "teal": rgb("#00BCD4"),
  "red": rgb("#FF5722"),
  "green": rgb("#72FC3F"),
  "celeste": gradient.linear(orange, purple)
)

#let thmtitle(t, color: rgb("#000000")) = {
  return text(weight: "semibold", fill: color)[#t]
}
#let thmname(t, color: rgb("#000000")) = {
  return text(fill: color)[(#t)]
}

#let thmtext(t, color: rgb("#000000")) = {
  let a = t.children
  if (a.at(0) == [ ]) {
    a.remove(0)
  }
  t = a.join()

  return text(font: "New Computer Modern", fill: color)[#t]
}

#let thmbase(
  identifier,
  head,
  ..blockargs,
  supplement: auto,
  padding: (top: 0.5em, bottom: 0.5em),
  namefmt: x => [(#x)],
  titlefmt: strong,
  bodyfmt: x => x,
  separator: [#h(0.1em).#h(0.2em) \ ],
  base: "heading",
  base_level: none,
) = {
  if supplement == auto {
    supplement = head
  }
  let boxfmt(name, number, body, title: auto, ..blockargs_individual) = {
    if not name == none {
      name = [ #namefmt(name)]
    } else {
      name = []
    }
    if title == auto {
      title = head
    }
    if not number == none {
      title += " " + number
    }
    title = titlefmt(title)
    body = bodyfmt(body)
    pad(
      ..padding,
      showybox(
        width: 100%,
        radius: 0.3em,
        breakable: true,
        padding: (top: 0em, bottom: 0em),
        ..blockargs.named(),
        ..blockargs_individual.named(),
        [#title#name#titlefmt(separator)#body],
      ),
    )
    
  }

  let auxthmenv = thmenv(
    identifier,
    base,
    base_level,
    boxfmt,
  ).with(supplement: supplement)

  return auxthmenv.with(numbering: none)
}

#let styled-thmbase = thmbase.with(titlefmt: thmtitle, namefmt: thmname, bodyfmt: thmtext)

#let builder-thmbox(color: rgb("#000000"), ..builderargs) = styled-thmbase.with(
  titlefmt: thmtitle.with(color: color.darken(30%)),
  bodyfmt: thmtext.with(color: color.darken(70%)),
  namefmt: thmname.with(color: color.darken(30%)),
  frame: (
    body-color: color.lighten(92%),
    border-color: color.darken(10%),
    thickness: 1.5pt,
    inset: 1.2em,
    radius: 0.3em,
  ),
  ..builderargs,
)
#let builder-thmbox_celeste(color: rgb("#000000"), ..builderargs) = styled-thmbase.with(
  titlefmt: thmtitle.with(color: color.darken(30%)),
  bodyfmt: thmtext.with(color: color.darken(70%)),
  namefmt: thmname.with(color: color.darken(30%)),
  frame: (
    body-color: color.lighten(92%),
    border-color: gradient.linear(orange, purple),
    thickness: 1.5pt,
    inset: 1.2em,
    radius: 0.3em,
  ),
  ..builderargs,
)

#let builder-thmline(color: rgb("#000000"), ..builderargs) = styled-thmbase.with(
  titlefmt: thmtitle.with(color: color.darken(30%)),
  bodyfmt: thmtext.with(color: color.darken(70%)),
  namefmt: thmname.with(color: color.darken(30%)),
  frame: (
    body-color: color.lighten(92%),
    border-color: color.darken(10%),
    thickness: (left: 2pt),
    inset: 1.2em,
    radius: 0em,
  ),
  ..builderargs,
)

#let problem-style = builder-thmbox(color: colors.red, shadow: (offset: (x: 2pt, y: 2pt), color: luma(70%)))
#let problem = problem-style("", "Problem")

#let idea-style = builder-thmbox(color: colors.blue, shadow: (offset: (x: 3pt, y: 3pt), color: luma(70%)))
#let idea = idea-style("", "Idea")

#let definition-style = builder-thmline(color: colors.teal)
#let definition = definition-style("", "Definition")

#let great-style = builder-thmbox_celeste(color: purple, shadow: (offset: (x: 3pt, y: 3pt), color: luma(70%)))
#let great = great-style("", rainbow("Great!", color_from: orange, color_to: purple, angle: 0deg))

#let example-style = builder-thmline(color: colors.red)
#let example = example-style("", "Example").with(numbering: none)

#let abstract_(body, name: none) = {
  thmtitle[ABSTRACT]
  if name != none {
    [ #thmname[#name]]
  }
  thmtitle[.]
  body
  h(1fr)
}
