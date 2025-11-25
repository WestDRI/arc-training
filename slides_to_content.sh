# for file in $(fd slides.qmd$)
# do
#     newname=$(echo $file | sed "s/slides/content/")
#     cp $file $newname
# done

slides_to_content() {
    newname=$(echo $1 | sed "s/slides/content/")
    cp $1 $newname

    sed -i -z 's/ ## {.center}\n\n//g' $newname
    sed -i 's/ {.center}//g' $newname
    sed -i -z 's/. . .\n\n//g' $newname

    sed -i '/^frontpic:/d' $newname
    sed -i '/^frontpicwidth:/d' $newname
    sed -i '/^frontpicmargintop:/d' $newname
    sed -i '/^frontpicmarginbottom:/d' $newname
    sed -i '/^frontpicborderradius:/d' $newname
    sed -i '/^noshadow: noshadow/d' $newname
    sed -i '/^frontlogo: \/img\/logo_sfudrac.png/d' $newname
    sed -i '/^date:/d' $newname
    sed -i '/^date-format: long/d' $newname
    sed -i '/^execute:/d' $newname
    sed -i '/^  freeze: auto/d' $newname
    sed -i '/^  cache: true/d' $newname
    sed -i '/^  error: true/d' $newname
    sed -i '/^  echo: true/d' $newname
    sed -i '/^format:/d' $newname
    sed -i '/^  revealjs:/d' $newname
    sed -i '/^    embed-resources: true/d' $newname
    sed -i '/^    theme: \[default,/d' $newname
    sed -i '/^    logo: \/img\/favicon_sfudrac.png/d' $newname
    sed -i '/^    highlight-style:/d' $newname
    sed -i '/^    code-line-numbers: false/d' $newname
    sed -i '/^    code-overflow: wrap/d' $newname
    sed -i '/^    template-partials:/d' $newname
    sed -i '/title-slide.html/d' $newname
    sed -i '/^    pointer:/d' $newname
    sed -i '/^      color: "#b5111b"/d' $newname
    sed -i '/^      pointerSize: 32/d' $newname
    sed -i '/^    link-external-newwindow: true/d' $newname
    sed -i '/^    footer: <a href=/d' $newname
    sed -i '/^    auto-stretch: false/d' $newname
    sed -i '/^revealjs-plugins:/d' $newname
    sed -i '/^  - pointer/d' $newname

    sed -i '0,/^---/! {0,/^---/ s/^---/---\n\n:::{.def}\n\n*Content from [the webinar slides](wb_xxx_slides.qmd) for easier browsing.*\n\n:::/}' $newname
}
