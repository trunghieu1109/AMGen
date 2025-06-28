async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    """
    [Stage 0: Extract Molecular Structures and Key Features]
    [Objective] 
    - Extract and clearly define the molecular structures of each given molecule: triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone.
    - Gather or construct their 3D geometries, identify key structural features, and note any known or reported symmetry elements from literature or databases.
    [Agent Collaborations]
    - Chain-of-Thought (CoT) to ensure detailed stepwise extraction and clear definition.
    """
    cot_instruction = "Sub-task 1: Extract and define the molecular structures and key features of the four given molecules, including 3D geometry and known symmetry elements from literature or databases."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting molecular structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 1: Analyze Symmetry Elements and Determine Point Groups]
    [Objective] 
    - Analyze the molecular structures obtained in subtask_1 to identify all symmetry elements present in each molecule, such as rotation axes (Cn), mirror planes (σ), inversion centers (i), and improper rotation axes (Sn).
    - Determine the point group of each molecule with particular focus on detecting the presence or absence of C3h symmetry elements (a C3 principal axis and a horizontal mirror plane σh, without perpendicular C2 axes or vertical mirror planes).
    [Agent Collaborations]
    - Reflexion pattern to iteratively verify and refine symmetry element identification and point group assignment.
    """
    cot_reflect_instruction = "Sub-task 2: Analyze the molecular structures from Sub-task 1 to identify all symmetry elements and determine the point group of each molecule, focusing on C3h symmetry elements."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, analyzing symmetry elements, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                              "Critically evaluate the symmetry analysis and point group determination, and provide its limitations.", 
                                              i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining symmetry analysis, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Verify C3h Symmetry Elements and Confirm Point Group]
    [Objective] 
    - For each molecule, verify and document the presence or absence of the defining symmetry elements of the C3h point group based on the analysis in subtask_2.
    - Explicitly confirm the presence of a C3 axis and a horizontal mirror plane, and the absence of conflicting symmetry elements that would exclude C3h symmetry.
    [Agent Collaborations]
    - Chain-of-Thought (CoT) for explicit stepwise verification and documentation.
    """
    cot_instruction_3 = "Sub-task 3: Verify and document the presence or absence of C3h symmetry elements for each molecule based on Sub-task 2's analysis."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, verifying C3h symmetry elements, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Compare and Select Molecule with C3h Symmetry]
    [Objective] 
    - Compare the symmetry determinations from subtask_3 across all four molecules to identify which molecule(s) conform to the C3h symmetry criteria.
    - Summarize the key symmetry features that support or exclude each molecule from having C3h symmetry.
    - Select and clearly state the molecule from the given choices that exhibits C3h symmetry based on the comparative analysis.
    - Provide a concise justification referencing the identified symmetry elements and point group assignments.
    [Agent Collaborations]
    - Debate pattern to allow multiple agents to argue and refine the selection decision.
    """
    debate_instruction_4 = "Sub-task 4: Compare the verified symmetry elements from Sub-task 3 and select the molecule exhibiting C3h symmetry, providing justification."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
                input_infos_4.extend(all_answer4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting molecule with C3h symmetry, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on which molecule has C3h symmetry.", is_sub_task=True)
    agents.append(f"Final Decision agent on molecule selection, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
