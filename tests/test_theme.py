from footy.paths import CONFIG
from footy.viz.theme import apply_theme, load_brand


def test_brand_has_two_variants():
    dark, light = load_brand(), load_brand(variante="clara")
    assert (CONFIG / "brand.yaml").exists()
    assert dark.variante == "escura" and dark.colors["background"] == "#02070A"
    assert light.variante == "clara" and light.colors["background"] == "#FFFFFF"
    assert dark.colors["primary"] == light.colors["primary"] == "#FC734D"
    assert light.colors["primary_ink"] == "#D9532E" and dark.colors["primary_ink"] == "#FC734D"
    assert dark.com_variante("clara").colors == light.colors
    assert apply_theme(dark, "clara").variante == "clara"
