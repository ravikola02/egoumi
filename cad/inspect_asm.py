"""Load the UMI assembly via OCAF/XCAF, return list of (name, located cq.Shape)."""
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
import cadquery as cq
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.IFSelect import IFSelect_RetDone


def load_parts(path):
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    r = STEPCAFControl_Reader()
    r.SetNameMode(True)
    assert r.ReadFile(path) == IFSelect_RetDone
    r.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    out = []

    def name_of(lab):
        n = TDataStd_Name()
        if lab.FindAttribute(TDataStd_Name.GetID_s(), n):
            return n.Get().ToExtString()
        return "?"

    def rec(lab, loc, nm):
        if st.IsAssembly_s(lab):
            comps = TDF_LabelSequence()
            st.GetComponents_s(lab, comps)
            for i in range(1, comps.Length() + 1):
                c = comps.Value(i)
                ref = TDF_Label()
                st.GetReferredShape_s(c, ref)
                cl = st.GetLocation_s(c)
                rec(ref, loc.Multiplied(cl), name_of(ref))
        else:
            shp = st.GetShape_s(lab)
            out.append((nm, cq.Shape.cast(shp.Moved(loc))))

    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        rec(roots.Value(i), TopLoc_Location(), name_of(roots.Value(i)))
    return out


if __name__ == "__main__":
    parts = load_parts(os.path.join(_ROOT, "input", "vis_assembly.step"))
    for nm, s in parts:
        bb = s.BoundingBox()
        print(f"{nm:22s} vol={s.Volume():9.0f} x[{bb.xmin:7.1f},{bb.xmax:7.1f}] "
              f"y[{bb.ymin:7.1f},{bb.ymax:7.1f}] z[{bb.zmin:7.1f},{bb.zmax:7.1f}]")
