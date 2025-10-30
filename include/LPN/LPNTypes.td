#ifndef LPN_TYPES
#define LPN_TYPES

include "LPN/LPNDialect.td"

class LPN_Type<string name, string typeMnemonic> : TypeDef<LPN_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def LPN_TokenType : LPN_Type<"Token", "token"> {
  let parameters = (ins ArrayRefParameter<"StringAttr">:$properties);
  // ... rest of definition
}

def LPN_PlaceType : LPN_Type<"Place", "place"> {
  // ... definition
}

def LPN_TransitionType : LPN_Type<"Transition", "transition"> {
  // ... definition
}

#endif // LPN_TYPES