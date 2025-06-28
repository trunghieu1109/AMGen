async def forward_48(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Analyze the general mechanism, characteristics, and thermodynamic aspects of sigmatropic rearrangements, "
        "including the migration of terminal pi bonds into sigma bonds, with emphasis on Cope and Claisen rearrangements. "
        "Establish foundational knowledge of orbital symmetry rules (Woodward–Hoffmann) and typical stereochemical outcomes."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing sigmatropic rearrangements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Summarize the specific features, mechanisms, and stereochemical consequences of Cope and Claisen rearrangements, "
        "including common variants (e.g., aza-Cope, oxy-Cope, enyne Cope), to prepare for detailed mechanistic analysis of the given reactions."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, summarizing Cope and Claisen rearrangements, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_3A = (
        "Sub-task 3A: Perform a detailed mechanistic analysis of reaction A (1,1-dimethoxyethan-1-amine + but-3-en-2-ol + H+ + heat). "
        "Generate explicit molecular structures (e.g., SMILES), enumerate all plausible sigmatropic and alternative pericyclic pathways including aza-Cope and Claisen variants, "
        "evaluate orbital symmetry and thermodynamic favorability, and predict the most chemically plausible product with stereochemical details. "
        "Use Self-Consistency Chain-of-Thought to generate multiple independent reasoning chains and aggregate results."
    )
    cot_agents_3A = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3A = []
    thinkingmapping_3A = {}
    answermapping_3A = {}
    subtask_desc3A = {
        "subtask_id": "subtask_3A",
        "instruction": cot_sc_instruction_3A,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3A, answer3A = await cot_agents_3A[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3A, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3A[i].id}, analyzing reaction A, thinking: {thinking3A.content}; answer: {answer3A.content}")
        possible_answers_3A.append(answer3A.content)
        thinkingmapping_3A[answer3A.content] = thinking3A
        answermapping_3A[answer3A.content] = answer3A
    answer3A_content = Counter(possible_answers_3A).most_common(1)[0][0]
    thinking3A = thinkingmapping_3A[answer3A_content]
    answer3A = answermapping_3A[answer3A_content]
    sub_tasks.append(f"Sub-task 3A output: thinking - {thinking3A.content}; answer - {answer3A.content}")
    subtask_desc3A["response"] = {"thinking": thinking3A, "answer": answer3A}
    logs.append(subtask_desc3A)
    print("Step 3A: ", sub_tasks[-1])
    
    cot_sc_instruction_3B = (
        "Sub-task 3B: Perform a detailed mechanistic analysis of reaction B ((3R,4S)-3,4-dimethylhexa-1,5-diyne + heat). "
        "Generate explicit molecular structures, enumerate all plausible sigmatropic and alternative pericyclic pathways including enyne Cope and other rearrangements, "
        "evaluate orbital symmetry and thermodynamic favorability, and predict the most chemically plausible product with stereochemical details. "
        "Use Self-Consistency Chain-of-Thought to generate multiple independent reasoning chains and aggregate results."
    )
    cot_agents_3B = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3B = []
    thinkingmapping_3B = {}
    answermapping_3B = {}
    subtask_desc3B = {
        "subtask_id": "subtask_3B",
        "instruction": cot_sc_instruction_3B,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3B, answer3B = await cot_agents_3B[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3B, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3B[i].id}, analyzing reaction B, thinking: {thinking3B.content}; answer: {answer3B.content}")
        possible_answers_3B.append(answer3B.content)
        thinkingmapping_3B[answer3B.content] = thinking3B
        answermapping_3B[answer3B.content] = answer3B
    answer3B_content = Counter(possible_answers_3B).most_common(1)[0][0]
    thinking3B = thinkingmapping_3B[answer3B_content]
    answer3B = answermapping_3B[answer3B_content]
    sub_tasks.append(f"Sub-task 3B output: thinking - {thinking3B.content}; answer - {answer3B.content}")
    subtask_desc3B["response"] = {"thinking": thinking3B, "answer": answer3B}
    logs.append(subtask_desc3B)
    print("Step 3B: ", sub_tasks[-1])
    
    cot_sc_instruction_3C = (
        "Sub-task 3C: Perform a detailed mechanistic analysis of reaction C (2-((vinyloxy)methyl)but-1-ene + heat). "
        "Generate explicit molecular structures, enumerate all plausible sigmatropic and alternative pericyclic pathways including Claisen and oxy-Cope variants, "
        "evaluate orbital symmetry and thermodynamic favorability, and predict the most chemically plausible product with stereochemical details. "
        "Use Self-Consistency Chain-of-Thought to generate multiple independent reasoning chains and aggregate results."
    )
    cot_agents_3C = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3C = []
    thinkingmapping_3C = {}
    answermapping_3C = {}
    subtask_desc3C = {
        "subtask_id": "subtask_3C",
        "instruction": cot_sc_instruction_3C,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3C, answer3C = await cot_agents_3C[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3C, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3C[i].id}, analyzing reaction C, thinking: {thinking3C.content}; answer: {answer3C.content}")
        possible_answers_3C.append(answer3C.content)
        thinkingmapping_3C[answer3C.content] = thinking3C
        answermapping_3C[answer3C.content] = answer3C
    answer3C_content = Counter(possible_answers_3C).most_common(1)[0][0]
    thinking3C = thinkingmapping_3C[answer3C_content]
    answer3C = answermapping_3C[answer3C_content]
    sub_tasks.append(f"Sub-task 3C output: thinking - {thinking3C.content}; answer - {answer3C.content}")
    subtask_desc3C["response"] = {"thinking": thinking3C, "answer": answer3C}
    logs.append(subtask_desc3C)
    print("Step 3C: ", sub_tasks[-1])
    
    cot_reflect_instruction_3_reflexion = (
        "Sub-task 3 Reflexion: Critically review and cross-validate the mechanistic conclusions and predicted products from subtasks 3A, 3B, and 3C. "
        "Ensure consistency with foundational knowledge from subtasks 1 and 2, verify stereochemical assignments, and confirm that no plausible sigmatropic or alternative pericyclic pathways have been overlooked or misclassified. "
        "Provide a detailed synthesis and correction if needed."
    )
    cot_agent_3_reflexion = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_reflexion = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_reflexion = self.max_round
    cot_inputs_3_reflexion = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3A, answer3A, thinking3B, answer3B, thinking3C, answer3C]
    subtask_desc3_reflexion = {
        "subtask_id": "subtask_3_reflexion",
        "instruction": cot_reflect_instruction_3_reflexion,
        "context": ["user query", "thinking and answer of subtasks 1, 2, 3A, 3B, 3C"],
        "agent_collaboration": "Reflexion"
    }
    thinking3_reflexion, answer3_reflexion = await cot_agent_3_reflexion(cot_inputs_3_reflexion, cot_reflect_instruction_3_reflexion, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_reflexion.id}, reviewing mechanistic conclusions, thinking: {thinking3_reflexion.content}; answer: {answer3_reflexion.content}")
    for i in range(N_max_3_reflexion):
        feedback, correct = await critic_agent_3_reflexion([taskInfo, thinking3_reflexion, answer3_reflexion],
                                                          "Please review the mechanistic conclusions and predicted products for consistency and completeness, and provide limitations.",
                                                          i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_reflexion.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_reflexion.extend([thinking3_reflexion, answer3_reflexion, feedback])
        thinking3_reflexion, answer3_reflexion = await cot_agent_3_reflexion(cot_inputs_3_reflexion, cot_reflect_instruction_3_reflexion, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_reflexion.id}, refining mechanistic conclusions, thinking: {thinking3_reflexion.content}; answer: {answer3_reflexion.content}")
    sub_tasks.append(f"Sub-task 3 Reflexion output: thinking - {thinking3_reflexion.content}; answer - {answer3_reflexion.content}")
    subtask_desc3_reflexion["response"] = {"thinking": thinking3_reflexion, "answer": answer3_reflexion}
    logs.append(subtask_desc3_reflexion)
    print("Step 3 Reflexion: ", sub_tasks[-1])
    
    debate_instruction_4 = (
        "Sub-task 4: Conduct a structured debate among multiple agents, each arguing for different multiple-choice options (A, B, C, D) based on the mechanistic and stereochemical evidence from subtask_3_reflexion. "
        "Use a referee agent to adjudicate and reach a consensus on the best matching option, employing a scoring rubric that considers stereochemistry, ring size, functional groups, and mechanistic consistency."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3_reflexion", "answer of subtask 3_reflexion"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3_reflexion, answer3_reflexion], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3_reflexion, answer3_reflexion] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating multiple-choice options, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the best matching multiple-choice option for reactions A, B, and C.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting best matching option, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5 = (
        "Sub-task 5: Select the final multiple-choice answer (A, B, C, or D) that best matches the predicted products for reactions A, B, and C, "
        "ensuring full alignment with sigmatropic rearrangement principles, stereochemical correctness, and reaction conditions."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, selecting final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
