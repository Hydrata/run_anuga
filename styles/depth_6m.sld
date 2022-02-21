<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" version="1.0.0" xmlns:ogc="http://www.opengis.net/ogc" xmlns:sld="http://www.opengis.net/sld" xmlns:gml="http://www.opengis.net/gml">
  <UserLayer>
    <sld:LayerFeatureConstraints>
      <sld:FeatureTypeConstraint/>
    </sld:LayerFeatureConstraints>
    <sld:UserStyle>
      <sld:Name>depth_6m</sld:Name>
      <sld:FeatureTypeStyle>
        <sld:Rule>
          <sld:RasterSymbolizer>
            <sld:ChannelSelection>
              <sld:GrayChannel>
                <sld:SourceChannelName>1</sld:SourceChannelName>
              </sld:GrayChannel>
            </sld:ChannelSelection>
            <sld:ColorMap type="ramp">
              <sld:ColorMapEntry color="#ffffff" quantity="0.00" label="0.00 m" opacity="0.0"/>
              <sld:ColorMapEntry color="#ffffff" quantity="0.05" label="0.05 m" opacity="0.0"/>
              <sld:ColorMapEntry color="#ffffff" quantity="0.10" label="0.10 m" opacity="0.6"/>
              <sld:ColorMapEntry color="#0014ff" quantity="0.20" label="0.20 m" opacity="0.8"/>
              <sld:ColorMapEntry color="#5191ff" quantity="0.50" label="0.50 m"/>
              <sld:ColorMapEntry color="#32f298" quantity="1" label="1.00 m"/>
              <sld:ColorMapEntry color="#a4fc3c" quantity="2" label="2.00 m"/>
              <sld:ColorMapEntry color="#eecf3a" quantity="3" label="3.00 m"/>
              <sld:ColorMapEntry color="#fb7e21" quantity="4" label="4.00 m"/>
              <sld:ColorMapEntry color="#d02f05" quantity="5" label="5.00 m"/>
              <sld:ColorMapEntry color="#7a0403" quantity="6" label="6.00 m"/>
            </sld:ColorMap>
          </sld:RasterSymbolizer>
        </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </UserLayer>
</StyledLayerDescriptor>
