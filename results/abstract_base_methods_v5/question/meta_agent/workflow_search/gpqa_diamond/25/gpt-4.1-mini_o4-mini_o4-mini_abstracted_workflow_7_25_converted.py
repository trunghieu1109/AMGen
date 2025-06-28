async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Understand the given chemical reaction: Cyclohexene + A → 8,8-diiodobicyclo[4.2.0]octan-7-one. Identify the structural and mechanistic requirements for reactant A to form the specified product, including the type of reaction (Diels–Alder) and key functional groups involved."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding reaction and requirements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    debate_instruction_2 = "Sub-task 2: Analyze the candidate reactants A (4,4-diiodocyclobut-2-en-1-one and 2,2-diiodoethen-1-one) for their structural features, reactivity, and compatibility with cyclohexene in the given Diels–Alder reaction. Determine which reactant A can feasibly yield the product based on mechanistic reasoning and output from Sub-task 1."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            input_infos_2 = [taskInfo, thinking1, answer1]
            thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing reactants A, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Make final decision on correct reactant A.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct reactant A, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction_3a = "Sub-task 3a: Evaluate each diene (1. 2,3-dimethylbuta-1,3-diene, 2. (2E,4E)-hexa-2,4-diene, 3. cyclopenta-1,3-diene, 4. (2Z,4Z)-hexa-2,4-diene) individually based on key criteria relevant to Diels–Alder reactions: s-cis conformation availability, electron-donating substituents, ring strain/rigidity, and stereochemistry."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, evaluating individual dienes, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_instruction_3b = "Sub-task 3b: Using Self-Consistency Chain-of-Thought, generate multiple independent reasoning paths to rank the dienes by reactivity from most reactive to least reactive, aggregating results to reach a consensus order based on criteria such as s-cis locking, electron donation, rigidity, and stereochemistry."
    N = self.max_sc
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, ranking dienes by reactivity, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    most_common_answer_3b = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[most_common_answer_3b]
    answer3b = answermapping_3b[most_common_answer_3b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    cot_instruction_4a = "Sub-task 4a: Restate and explicitly verify the reactant A choice from Sub-task 2 and the diene reactivity order from Sub-task 3b. Prepare a structured summary of these validated results for comparison against the multiple-choice options."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking2, answer2, thinking3b, answer3b], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, verifying reactant and diene order, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    debate_instruction_4b = "Sub-task 4b: Compare the validated reactant A and diene reactivity order against each multiple-choice option. Identify the option(s) that perfectly match both criteria without contradictions."
    debate_agents_4b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4b = self.max_round
    all_thinking4b = [[] for _ in range(N_max_4b)]
    all_answer4b = [[] for _ in range(N_max_4b)]
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": debate_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4b):
        for i, agent in enumerate(debate_agents_4b):
            input_infos_4b = [taskInfo, thinking4a, answer4a]
            thinking4b, answer4b = await agent(input_infos_4b, debate_instruction_4b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing options for consistency, thinking: {thinking4b.content}; answer: {answer4b.content}")
            all_thinking4b[r].append(thinking4b)
            all_answer4b[r].append(answer4b)
    final_decision_agent_4b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4b, answer4b = await final_decision_agent_4b([taskInfo] + all_thinking4b[-1] + all_answer4b[-1], "Sub-task 4b: Make final decision on matching multiple-choice option.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting consistent option, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    cot_reflect_instruction_4c = "Sub-task 4c: Implement a reflective cross-validation step to detect any inconsistencies between the integrated answer and prior subtasks. If inconsistencies are found, prompt re-evaluation or flag the issue before finalizing the answer."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4c = self.max_round
    cot_inputs_4c = [taskInfo, thinking2, answer2, thinking3b, answer3b, thinking4b, answer4b]
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_reflect_instruction_4c,
        "context": ["user query", "thinking and answer of subtask 2", "thinking and answer of subtask 3b", "thinking and answer of subtask 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, initial reflection on integrated answer, thinking: {thinking4c.content}; answer: {answer4c.content}")
    for i in range(N_max_4c):
        feedback, correct = await critic_agent_4c([taskInfo, thinking4c, answer4c], "Please review the integrated answer for logical consistency with prior subtasks and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_4c.extend([thinking4c, answer4c, feedback])
        thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, refining integrated answer, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    cot_instruction_5 = "Sub-task 5: Perform a final validation to ensure the selected multiple-choice answer components exactly match the validated reactant A and diene reactivity order from previous subtasks. Flag any logical inconsistencies before delivering the final answer."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking and answer of subtask 4c"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4c, answer4c], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, final validation of selected answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs