async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    stage_1_subtask_1a_instruction = (
        "Sub-task 1a: Explicitly identify and name the chemical reaction corresponding to the first equation (A + H2SO4 → 2,8-dimethylspiro[4.5]decan-6-one), "
        "including mechanistic details and typical reactants, by consulting standard chemical literature and known name reactions (e.g., pinacol rearrangement)."
    )
    cot_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": stage_1_subtask_1a_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1a, answer1a = await cot_agents_1a[i]([taskInfo], stage_1_subtask_1a_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1a[i].id}, identifying and naming reaction 1a, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[answer1a_content]
    answer1a = answermapping_1a[answer1a_content]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a["response"] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    stage_1_subtask_1b_instruction = (
        "Sub-task 1b: Explicitly identify and name the chemical reaction corresponding to the second equation (B + BuLi + H+ → 4-methyl-1-phenylpent-3-en-1-ol), "
        "including mechanistic details and typical reactants, by consulting standard chemical literature and known name reactions."
    )
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": stage_1_subtask_1b_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1b, answer1b = await cot_agents_1b[i]([taskInfo], stage_1_subtask_1b_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, identifying and naming reaction 1b, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    answer1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[answer1b_content]
    answer1b = answermapping_1b[answer1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b["response"] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    stage_1_subtask_2a_instruction = (
        "Sub-task 2a: Verify and cross-check the identified reactions from subtasks 1a and 1b against multiple literature sources and mechanistic reasoning to confirm the reaction names and typical starting materials, "
        "using a Self-Consistency Chain-of-Thought approach to generate and compare plausible mechanisms."
    )
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": stage_1_subtask_2a_instruction,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], stage_1_subtask_2a_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, verifying reactions 2a, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc_2a["response"] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    stage_1_subtask_2b_instruction = (
        "Sub-task 2b: Based on the verified reaction names and mechanisms, list the most plausible reactants A and B that correspond to the given products and reagents, "
        "providing chemical reasoning and literature references for each reactant choice."
    )
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": stage_1_subtask_2b_instruction,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a, answer2a], stage_1_subtask_2b_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, listing plausible reactants 2b, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc_2b["response"] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    stage_2_subtask_3_instruction = (
        "Sub-task 3: Analyze each provided choice (choice1 to choice4) by comparing the proposed reactants A and B with the verified reactants identified in subtask 2b, "
        "critically evaluating the chemical consistency, mechanistic plausibility, and structural correctness of each option."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2b, answer2b]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": stage_2_subtask_3_instruction,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, stage_2_subtask_3_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing choices, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], "Please review the analysis of choices and provide limitations or confirm correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, stage_2_subtask_3_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    stage_2_subtask_4_instruction = (
        "Sub-task 4: Perform a reflexive evaluation and debate on the analyses from subtask 3 to select the correct choice (A, B, C, or D), "
        "providing detailed justification for the selection based on mechanistic accuracy, literature support, and structural consistency with the given reactions."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": stage_2_subtask_4_instruction,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], stage_2_subtask_4_instruction, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, stage_2_subtask_4_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final choice, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the correct choice of reactants.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice decision, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs