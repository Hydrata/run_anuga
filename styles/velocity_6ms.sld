<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" version="1.0.0" xmlns:ogc="http://www.opengis.net/ogc" xmlns:sld="http://www.opengis.net/sld" xmlns:gml="http://www.opengis.net/gml">
  <UserLayer>
    <sld:LayerFeatureConstraints>
      <sld:FeatureTypeConstraint/>
    </sld:LayerFeatureConstraints>
    <sld:UserStyle>
      <sld:Name>velocity_6ms</sld:Name>
      <sld:FeatureTypeStyle>
        <sld:Rule>
          <sld:RasterSymbolizer>
            <sld:ColorMap type="ramp">
              <sld:ColorMapEntry color="#30123b" quantity="0" label="0.0 m/s" opacity="0"/>
              <sld:ColorMapEntry color="#011fff" quantity="0.5" label="0.5 m/s" opacity="0.2"/>
              <sld:ColorMapEntry color="#011fff" quantity="1" label="1.0 m/s" opacity="1"/>
              <sld:ColorMapEntry color="#28e2b3" quantity="2" label="2.0 m/s"/>
              <sld:ColorMapEntry color="#a4fc3c" quantity="3" label="3.0 m/s"/>
              <sld:ColorMapEntry color="#f5b633" quantity="4" label="4.0 m/s"/>
              <sld:ColorMapEntry color="#e0470c" quantity="5" label="5.0 m/s"/>
              <sld:ColorMapEntry color="#7a0403" quantity="6" label=">6.0 m/s"/>
            </sld:ColorMap>
          </sld:RasterSymbolizer>
        </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </UserLayer>
</StyledLayerDescriptor>
