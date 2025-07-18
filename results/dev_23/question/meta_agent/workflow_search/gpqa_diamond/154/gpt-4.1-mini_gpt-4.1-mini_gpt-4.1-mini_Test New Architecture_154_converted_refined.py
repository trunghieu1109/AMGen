async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract the given operators Px, Py, and Pz as explicit 3x3 matrices and the given state vector in the Pz eigenbasis. "
        "Verify the dimension of the Hilbert space by checking matrix sizes and eigenvalues of Pz to confirm the system is spin-1 (3-dimensional), not spin-1/2. "
        "Check Hermiticity of the operators and confirm the state vector is normalized. "
        "This step explicitly addresses the previous failure of misidentifying the system dimension and operator forms by requiring concrete verification before proceeding."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Using the verified 3x3 Pz operator and the normalized 3-component state vector from Subtask 1, "
        "explicitly compute the expectation values <Pz> = <psi|Pz|psi> and <Pz^2> = <psi|Pz^2|psi> by performing the full matrix-vector multiplications and inner products. "
        "Avoid any spin-1/2 simplifications or formulas. This subtask directly addresses the previous error of applying spin-1/2 formulas to a spin-1 system and ensures numerical correctness of these key quantities."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Calculate the uncertainty ΔPz = sqrt(<Pz^2> - <Pz>^2) using the expectation values obtained in Subtask 2. "
        "Then, compare the computed uncertainty with the given multiple-choice options to identify the correct answer. "
        "Include a sanity check to confirm that the uncertainty value is physically reasonable given the spin-1 system and the operators involved. "
        "This subtask prevents repeating the previous mistake of incorrect uncertainty calculation by relying solely on rigorously computed values and explicit verification."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1.get('thinking', ''), results1.get('answer', ''), results2.get('thinking', ''), results2.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3.get('thinking', ''), results3.get('answer', ''))
    return final_answer, logs
