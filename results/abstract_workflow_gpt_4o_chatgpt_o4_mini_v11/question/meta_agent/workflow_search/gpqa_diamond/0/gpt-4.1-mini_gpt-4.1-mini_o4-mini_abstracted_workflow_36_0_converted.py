async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    """
    [Stage 1: Obtain and characterize molecular 3D structures]
    [Objective] 
    - For each molecule (quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone), obtain or generate detailed 3D structural representations including SMILES, InChI, or coordinate sketches.
    - Extract key geometric parameters such as bond lengths, bond angles, dihedral angles, and spatial orientation of substituents to enable rigorous symmetry analysis.
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) agent to systematically analyze and generate detailed structural data for each molecule.
    """
    
    cot_instruction_1 = (
        "Sub-task 1: For each molecule (quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, "
        "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone), obtain or generate detailed 3D structural representations including SMILES, InChI, or coordinate sketches. "
        "Extract key geometric parameters such as bond lengths, bond angles, dihedral angles, and spatial orientation of substituents to enable rigorous symmetry analysis. "
        "Provide a comprehensive and standardized structural dataset for all molecules."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, generating detailed 3D structures and geometric parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Define and apply C3h symmetry criteria]
    [Objective] 
    - Review and clearly define the criteria for C3h point group symmetry, including identification of the principal C3 rotational axis and the horizontal mirror plane (σh).
    - Using the detailed 3D structural data from subtask_1 and the C3h symmetry criteria, perform a systematic symmetry element analysis for each molecule.
    - Verify the presence and orientation of a C3 principal axis, check for a horizontal mirror plane, and evaluate substituent conformations that affect σh symmetry.
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought (SC-CoT) to explore multiple symmetry analysis scenarios and ensure robust criteria application.
    """
    cot_sc_instruction_2 = (
        "Sub-task 2: Review and define explicit criteria for C3h point group symmetry, including the principal C3 rotational axis and horizontal mirror plane (σh). "
        "Then, using the 3D structural data from Sub-task 1, perform a systematic symmetry element analysis for each molecule, verifying presence and orientation of C3 axis and σh plane, and evaluate substituent conformations affecting σh symmetry. "
        "Provide detailed documentation of symmetry elements found and any deviations negating C3h symmetry."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing C3h symmetry criteria and applying to molecules, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Synthesize symmetry analysis and assign point groups]
    [Objective] 
    - Compare and synthesize the symmetry analysis results from Stage 2 for all molecules.
    - Identify which molecule(s) satisfy all C3h symmetry criteria based on documented symmetry elements and conformational assessments.
    - Prepare a structured summary listing each molecule, its key symmetry elements, and final point group assignment.
    - Provide a clear, justified final answer to the original question.
    [Agent Collaborations]
    - Use Debate agents to discuss and converge on the final classification and answer.
    """
    debate_instruction_3 = (
        "Sub-task 3: Based on the outputs of Sub-tasks 1 and 2, synthesize the symmetry analysis results for all molecules. "
        "Identify which molecule(s) satisfy all C3h symmetry criteria, prepare a structured summary of symmetry elements and point group assignments, "
        "and provide a clear, justified final answer to the question: which molecule has C3h symmetry?"
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            input_infos_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
            if r > 0:
                input_infos_3 += all_thinking3[r-1] + all_answer3[r-1]
            thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, synthesizing symmetry analysis and final classification, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make a final decision on which molecule has C3h symmetry.", is_sub_task=True)
    agents.append(f"Final Decision agent on final classification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
