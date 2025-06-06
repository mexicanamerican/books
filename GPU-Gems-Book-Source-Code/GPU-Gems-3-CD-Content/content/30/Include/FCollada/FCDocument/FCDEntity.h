/*
    Copyright (C) 2005-2007 Feeling Software Inc.
    MIT License: http://www.opensource.org/licenses/mit-license.php
*/
/*
	Based on the FS Import classes:
	Copyright (C) 2005-2006 Feeling Software Inc
	Copyright (C) 2005-2006 Autodesk Media Entertainment
	MIT License: http://www.opensource.org/licenses/mit-license.php
*/

/**
	@file FCDEntity.h
	This file contains the FCDEntity class.
*/

#ifndef _FCD_ENTITY_H_
#define _FCD_ENTITY_H_

#ifndef __FCD_OBJECT_H_
#include "FCDocument/FCDObject.h"
#endif // __FCD_OBJECT_H_

class FCDocument;
class FCDAsset;
class FCDExtra;

typedef fm::pvector<FCDAsset> FCDAssetList;
typedef fm::pvector<const FCDAsset> FCDAssetConstList;

/**
	A COLLADA entity.

	A COLLADA entity is an object contained within a COLLADA library.
	As such, it is based on the FCDObjectWithId class so that it
	can be accessed by other entities, such as the scene graph.

	The entity adds to the FCDObjectWithId class: a name,
	an extra tree and an optional note, as well as a way
	to identity the type of the entity, in order to up-cast it
	to its correct class.

	@ingroup FCDocument
*/

class FCOLLADA_EXPORT FCDEntity : public FCDObjectWithId
{
public:
	/** The types of entity classes.
		Each type corresponds directly to one class that contains the
		FCDEntity class as a parent, so you can up-cast FCDEntity pointers. */
	enum Type
	{
		ENTITY, /**< A generic entity (FCDEntity). Should never be used. */
		ANIMATION, /**< An animation (FCDAnimation). */
		ANIMATION_CLIP, /**< An animation clip (FCDAnimationClip). */
		CAMERA, /**< A camera (FCDCamera). */
		LIGHT, /**< A light (FCDLight). */
		IMAGE, /**< An image (FCDImage). */
		MATERIAL, /**< A visual material definition (FCDMaterial). */
		EFFECT, /**< An effect definition (FCDEffect). */
		GEOMETRY, /**< A geometric object (FCDGeometry). Includes splines and meshes. */
		CONTROLLER, /**< A geometric controller (FCDController). Includes skins and morphers. */
		SCENE_NODE, /**< A visual scene node (FCDSceneNode). */
		PHYSICS_RIGID_CONSTRAINT, /**< A physics rigid constraint (FCDPhysicsRigidConstraint). */
		PHYSICS_MATERIAL, /**< A physics material definiton (FCDPhysicsMaterial). */
		PHYSICS_RIGID_BODY, /**< A physics rigid body (FCDPhysicsRigidBody). */
		PHYSICS_SHAPE, /**< A physics shape (FCDPhysicsShape). */
		PHYSICS_ANALYTICAL_GEOMETRY, /**< A physics analytical geometric object (FCDPhysicsAnalyticalGeometry). */
		PHYSICS_MODEL, /**< A physics model (FCDPhysicsModel). */
		PHYSICS_SCENE_NODE, /**< A physics scene node (FCDPhysicsScene). */
		FORCE_FIELD, /**< A physics force field (FCDForceField). */
		EMITTER, /**< A generic emitter for sound and particles (FCDEmitter) - Premium feature. */
	};

private:
	DeclareObjectType(FCDObjectWithId);
	fstring name;

	// Extra information for the entity.
	FUObjectRef<FCDExtra> extra;

	// Asset information for the entity.
	FUObjectRef<FCDAsset> asset;

	// Maya and Max both support custom strings for objects.
	fstring note;

	// Deprecated ColladaMaya post-processing information.
	StringList postCmds;

public:
	/** Constructor: do not use directly.
		Instead, create objects of the up-classes.
		@param document The COLLADA document that owns the entity.
		@param baseId The prefix COLLADA id to be used if no COLLADA id is provided. */
	FCDEntity(FCDocument* document, const char* baseId = "GenericEntity");

	/** Destructor. */
	virtual ~FCDEntity();

	/** Retrieves the entity class type for an entity.
		You can use the entity class type of up-cast an entity pointer
		to the correct up-class.
		This function should be overwritten by all up-classes.
		@return The entity class type. */
	virtual Type GetType() const { return ENTITY; }

	/** Retrieves the name of the entity.
		This value has no direct use in COLLADA but is useful
		to track the user-friendly name of an entity.
		@return The name. */
	const fstring& GetName() const { return name; }

	/** Sets the name of the entity.
		This value has no direct use in COLLADA but is useful
		to track the user-friendly name of an entity.
		@param _name The name. */
	void SetName(const fstring& _name);

	/** Retrieves the extra information tree for this entity.
		The prefered way to save extra information in FCollada is at
		the entity level. Use this extra information tree to store
		any information you want exported and imported back.
		@return The extra information tree. */
	FCDExtra* GetExtra() { return extra; }
	const FCDExtra* GetExtra() const { return extra; } /**< See above. */

	/** Retrieves the asset information for this entity.
		In the non-const version, calling this function will
		create an asset structure. If you are interested in a specific asset-type
		information, such as up_axis and units, please use the
		FCDEntity::GetHierarchicalAssets function.
		@return The asset information structure. */
	FCDAsset* GetAsset();
	inline const FCDAsset* GetAsset() const { return asset; } /**< See above. */

	/** Retrieves the asset information structures that affect
		this entity in its hierarchy.
		@param assets A list of asset information structures to fill in. */
	inline void GetHierarchicalAssets(FCDAssetList& assets) { GetHierarchicalAssets(*(FCDAssetConstList*) &assets); }
	virtual void GetHierarchicalAssets(FCDAssetConstList& assets) const; /**< See above. */

	/** Retrieves whether the entity has a user-defined note.
		This value is a simpler way, than the extra tree, to store
		user-defined information that does not belong in COLLADA.
		@return Whether the entity has an user-defined note. */
	bool HasNote() const { return !note.empty(); }

	/** Retrieves the user-defined note for this entity.
		This value is a simpler way, than the extra tree, to store
		user-defined information that does not belong in COLLADA.
		@return The user-defined note. */
	const fstring& GetNote() const { return note; }

	/** Sets the user-defined note for this entity.
		This value is a simpler way, than the extra tree, to store
		user-defined information that does not belong in COLLADA.
		@param _note The user-defined note. */
	void SetNote(const fstring& _note) { note = _note; SetDirtyFlag(); }

	/** Retrieves the child entity that has the given COLLADA id.
		This function is only useful for entities that are hierarchical:
		visual/physics scene nodes and animations.
		@param daeId A COLLADA id.
		@return The child entity with the given id. This pointer will be NULL
			if no child entity matches the given id. */
	virtual FCDEntity* FindDaeId(const fm::string& daeId);

	/** Copies the entity information into a clone.
		All the overwriting functions of this function should call this function
		to copy the COLLADA id and the other entity-level information.
		All the up-classes of this class should implement this function.
		The cloned entity may reside in another document.
		@param clone The empty clone. If this pointer is NULL, a new entity
			will be created and you will need to release the returned pointer manually.
		@param cloneChildren Whether to recursively clone this entity's children.
		@return The clone. */
	virtual FCDEntity* Clone(FCDEntity* clone = NULL, bool cloneChildren = false) const;

	/** [INTERNAL] Reads in the entity from a given COLLADA XML tree node.
		This function should be overwritten by all up-classes.
		@param entityNode The COLLADA XML tree node.
		@return The status of the import. If the status is 'false',
			it may be dangerous to extract information from the entity.*/
	virtual bool LoadFromXML(xmlNode* entityNode);

	/** [INTERNAL] Writes out the entity to the given COLLADA XML tree node.
		This function should be overwritten by all up-classes.
		@param parentNode The COLLADA XML parent node in which to insert the entity.
		@return The created element XML tree node. */
	virtual xmlNode* WriteToXML(xmlNode* parentNode) const;

protected:
	/** [INTERNAL] Writes out the top entity XML node for the entity.
		This function should be used by all up-classes within the
		WriteToXML overwriting function to create the top XML node,
		as it will write out the name and COLLADA id of the entity.
		@param parentNode The COLLADA XML parent node in which to insert the entity.
		@param nodeName The COLLADA XML node name for the top entity XML node.
		@return The created element XML tree node. */
	xmlNode* WriteToEntityXML(xmlNode* parentNode, const char* nodeName, bool writeId=true) const;

	/** [INTERNAL] Writes out the extra information for the entity.
		This function should be used by all up-classes within the
		WriteToXML overwriting function, at the very end, to write
		the user-defined note and the extra tree to the COLLADA document.
		@param entityNode The created element XML tree node returned
			by the WriteToEntityXML function. */
	virtual void WriteToExtraXML(xmlNode* entityNode) const;
};

#endif // _FCD_ENTITY_H_

