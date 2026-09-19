// Stand-in for Geant4's G4Exception.hh: the severity enumeration and the free function the overlay
// glue calls. The definition lives in overlay_check.cc.
#ifndef G4EXCEPTION_HH
#define G4EXCEPTION_HH

enum G4ExceptionSeverity {
  FatalException,
  FatalErrorInArgument,
  RunMustBeAborted,
  EventMustBeAborted,
  JustWarning
};

void G4Exception(const char* originOfException, const char* exceptionCode, G4ExceptionSeverity severity,
                 const char* description);

#endif
