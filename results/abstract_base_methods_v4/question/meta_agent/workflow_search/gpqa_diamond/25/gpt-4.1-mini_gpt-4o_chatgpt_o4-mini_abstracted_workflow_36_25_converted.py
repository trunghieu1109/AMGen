async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Analyze each given diene (1. 2,3-dimethylbuta-1,3-diene, 2. (2E,4E)-hexa-2,4-diene, "
        "3. cyclopenta-1,3-diene, 4. (2Z,4Z)-hexa-2,4-diene) individually to identify and characterize key structural and electronic factors affecting their reactivity, "
        "including conjugation extent, substitution pattern, ring strain, and conformational preferences (e.g., s-cis vs s-trans). "
        "Provide detailed mechanistic reasoning and justify how each factor influences reactivity."
    )
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking1 = [[] for _ in range(N_max_1)]
    all_answer1 = [[] for _ in range(N_max_1)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                input_infos_1 = [taskInfo]
            else:
                input_infos_1 = [taskInfo] + all_thinking1[r-1] + all_answer1[r-1]
            thinking1, answer1 = await agent(input_infos_1, cot_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing dienes' structural and electronic factors, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking1[-1] + all_answer1[-1], "Sub-task 1: Make final decision on the detailed analysis of dienes' reactivity factors.", is_sub_task=True)
    agents.append(f"Final Decision agent on diene analysis, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2a = (
        "Sub-task 2a: Perform a detailed mechanistic analysis of the reaction Cyclohexene + A → 8,8-diiodobicyclo[4.2.0]octan-7-one. "
        "Identify the reaction type (e.g., Diels–Alder, [2+2] cycloaddition, halogenation) based on the product structure, electron count, and typical reactivity patterns. "
        "Discuss possible mechanisms and justify the most plausible one."
    )
    debate_agents_2a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2a = self.max_round
    all_thinking2a = [[] for _ in range(N_max_2a)]
    all_answer2a = [[] for _ in range(N_max_2a)]
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2a):
        for i, agent in enumerate(debate_agents_2a):
            if r == 0:
                input_infos_2a = [taskInfo, thinking1, answer1]
            else:
                input_infos_2a = [taskInfo, thinking1, answer1] + all_thinking2a[r-1] + all_answer2a[r-1]
            thinking2a, answer2a = await agent(input_infos_2a, cot_instruction_2a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mechanistic analysis, thinking: {thinking2a.content}; answer: {answer2a.content}")
            all_thinking2a[r].append(thinking2a)
            all_answer2a[r].append(answer2a)
    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2a, answer2a = await final_decision_agent_2a([taskInfo, thinking1, answer1] + all_thinking2a[-1] + all_answer2a[-1], "Sub-task 2a: Make final decision on the reaction mechanism.", is_sub_task=True)
    agents.append(f"Final Decision agent on mechanistic analysis, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = (
        "Sub-task 2b: Based on the identified reaction mechanism from Sub-task 2a, deduce the roles of cyclohexene and candidate reactants A (4,4-diiodocyclobut-2-en-1-one and 2,2-diiodoethen-1-one) in the reaction. "
        "Justify the selection of the correct reactant A with respect to electronic structure, ring size, and stereochemical outcomes. "
        "Use Self-Consistency Chain-of-Thought to generate and evaluate multiple plausible reactant hypotheses."
    )
    N_sc_2b = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2b)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2b):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking1, answer1, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, deducing reactant A roles, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_3 = (
        "Sub-task 3: Refine and establish the reactivity order of the given dienes from most reactive to least reactive by integrating mechanistic insights from Sub-tasks 1 and 2b. "
        "Consider factors such as activation energies, HOMO–LUMO interactions, and conformational accessibility relevant to the identified reaction pathway. "
        "Use Reflexion with iterative critic feedback to improve the reasoning and final ranking."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2b, answer2b]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining diene reactivity order, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Please review the refined diene reactivity order and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining diene reactivity order, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = (
        "Sub-task 4: Match the deduced correct reactant A (from Sub-task 2b) and the refined diene reactivity sequence (from Sub-task 3) "
        "against the provided multiple-choice options to select the correct answer, ensuring consistency with mechanistic reasoning and reaction conditions. "
        "Use Debate pattern with multiple agents and rounds to avoid premature consensus."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                input_infos_4 = [taskInfo, thinking2b, answer2b, thinking3, answer3]
            else:
                input_infos_4 = [taskInfo, thinking2b, answer2b, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching reactant and reactivity order, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the correct reactant A and diene reactivity sequence.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting correct reactant and reactivity order, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
