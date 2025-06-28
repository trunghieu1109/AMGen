async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 1: Collect and define molecular structures with detailed 2D/3D data and formulas
    cot_instruction_1 = (
        "Subtask 1: Collect and clearly define the molecular structures, including detailed 2D or 3D "
        "representations (e.g., Cartesian coordinates or planar diagrams), and chemical formulas of each given molecule: "
        "triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. "
        "Ensure hydrogen counts and substituent details are explicitly included to support symmetry analysis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, collecting molecular structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    # Stage 1: Research and summarize C3h symmetry elements and identification criteria
    cot_instruction_2 = (
        "Subtask 2: Research and summarize the concept of C3h molecular symmetry, explicitly listing all symmetry elements involved "
        "(E, C3, C3^2, σ_h, and absence of σ_v), and describe the criteria and methods for identifying these elements in molecular structures."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, summarizing C3h symmetry, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    # Stage 2: Enumerate candidate vertical mirror planes (σ_v) for each molecule using detailed structural data
    cot_sc_instruction_3 = (
        "Subtask 3: Enumerate all candidate vertical mirror planes (σ_v) for each molecule using the detailed structural data from Subtask 1, "
        "generating explicit hypotheses of possible σ_v planes based on molecular geometry and substituent positions."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, enumerating candidate σ_v planes, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 2: Test each enumerated σ_v candidate plane against atomic positions and substituent symmetries
    cot_sc_instruction_4 = (
        "Subtask 4: Test each enumerated σ_v candidate plane against the actual atomic positions and substituent symmetries for each molecule "
        "to determine which σ_v planes, if any, are valid symmetry elements. Document which planes pass or fail the test."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, testing candidate σ_v planes, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Subtask 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    # Stage 2: Identify and verify presence of other C3h symmetry elements (E, C3, C3^2, σ_h)
    cot_instruction_5 = (
        "Subtask 5: Identify and verify the presence of other C3h symmetry elements (E, C3, C3^2, σ_h) for each molecule using the structural data "
        "and the criteria from Subtask 2, ensuring a comprehensive symmetry element inventory for each molecule."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, cot_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying other C3h elements, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Subtask 5: Make final decision on presence of other C3h symmetry elements.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on other C3h elements, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    # Stage 2: Integrate results from subtasks 4 and 5 to determine molecules with C3h symmetry using self-consistency or debate
    cot_sc_instruction_6 = (
        "Subtask 6: Integrate the results from Subtasks 4 and 5 to conclusively determine which molecule(s) exhibit C3h symmetry by confirming presence of C3 and σ_h elements "
        "and absence of any σ_v planes. Use a self-consistency approach by performing at least two independent symmetry assessments or a debate-style cross-validation to ensure robustness of the classification."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking4, answer4, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, integrating symmetry results, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    answer6_content = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[answer6_content]
    answer6 = answermapping_6[answer6_content]
    sub_tasks.append(f"Subtask 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    # Stage 3: Select molecule(s) with confirmed C3h symmetry and provide final answer in multiple-choice format
    cot_instruction_7 = (
        "Subtask 7: Select the molecule(s) with confirmed C3h symmetry from the given choices and provide the final answer strictly in the required multiple-choice format (A, B, C, or D), "
        "supported by a concise summary of the symmetry analysis. Enforce output format validation to return only the letter."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    final_answer_content = answer7.content.strip()
    if final_answer_content not in ["A", "B", "C", "D"]:
        final_answer_content = "Answer format invalid, expected one of A, B, C, or D."
    agents.append(f"CoT agent {cot_agent_7.id}, final answer selection, thinking: {thinking7.content}; answer: {final_answer_content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {thinking7.content}; answer - {final_answer_content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
