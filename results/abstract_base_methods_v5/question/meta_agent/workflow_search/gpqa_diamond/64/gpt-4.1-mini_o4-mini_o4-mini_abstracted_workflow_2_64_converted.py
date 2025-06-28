async def forward_64(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = (
        "Sub-task 1a: Identify and list all possible reaction types and conditions involved in the first reaction (vinylspiro[3.5]non-5-en-1-ol + THF, KH, H+), "
        "explicitly considering the effect of heat (e.g., heating to ~100 6C), base (KH), and acidic workup (H+), with emphasis on the anionic oxy-Cope rearrangement and retention of alkene conjugation."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, identifying reaction types and conditions for first reaction, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = (
        "Sub-task 1b: Generate multiple plausible mechanistic pathways for the first reaction, including stereochemical outcomes and the effect of heat on product formation, "
        "focusing on whether the product is an unsaturated enone or a saturated ketone. Use the reaction types and conditions identified in Sub-task 1a."
    )
    N_sc_1b = self.max_sc
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1b)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1b):
        thinking_1b, answer_1b = await cot_agents_1b[i]([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, generating mechanistic pathways for first reaction, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
        possible_answers_1b.append(answer_1b.content)
        thinkingmapping_1b[answer_1b.content] = thinking_1b
        answermapping_1b[answer_1b.content] = answer_1b
    answer_1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking_1b = thinkingmapping_1b[answer_1b_content]
    answer_1b = answermapping_1b[answer_1b_content]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = (
        "Sub-task 1c: Predict and rationalize the major product(s) of the first reaction based on the mechanistic pathways from Sub-task 1b, "
        "explicitly validating the product structure against known oxy-Cope rearrangement literature, ensuring the product retains alkene conjugation and is (E)-bicyclo[5.3.1]undec-1(11)-en-4-one rather than a fully saturated ketone."
    )
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo, thinking_1b, answer_1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, predicting major product of first reaction, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_instruction_2a = (
        "Sub-task 2a: Analyze the second reaction ((E)-pent-2-en-1-ol + acetyl bromide with LDA) to identify the reaction mechanism, intermediates, "
        "and influence of the strong base (LDA) and reaction medium on product formation."
    )
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_2a, answer_2a = await cot_agent_2a([taskInfo], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, analyzing second reaction mechanism, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = (
        "Sub-task 2b: Predict the major product(s) of the second reaction based on mechanistic analysis from Sub-task 2a, "
        "including the nature of the product (e.g., lithium salt vs acid) and stereochemical considerations."
    )
    N_sc_2b = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2b)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2b):
        thinking_2b, answer_2b = await cot_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, predicting major product of second reaction, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    answer_2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[answer_2b_content]
    answer_2b = answermapping_2b[answer_2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_sc_instruction_3 = (
        "Sub-task 3: Perform a self-consistency chain-of-thought (SC CoT) validation and reflexion on the predicted major product of reaction A from Sub-task 1c, "
        "comparing alternative plausible products and confirming the correct product structure with mechanistic and literature support."
    )
    N_sc_3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_1c, answer_1c], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, validating major product of reaction A, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3.content)
        thinkingmapping_3[answer_3.content] = thinking_3
        answermapping_3[answer_3.content] = answer_3
    answer_3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking_3 = thinkingmapping_3[answer_3_content]
    answer_3 = answermapping_3[answer_3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = (
        "Sub-task 4: Perform a self-consistency chain-of-thought (SC CoT) validation and reflexion on the predicted major product of reaction B from Sub-task 2b, "
        "confirming the product identity and form (lithium salt vs acid) with mechanistic rationale."
    )
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking_4, answer_4 = await cot_agents_4[i]([taskInfo, thinking_2b, answer_2b], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, validating major product of reaction B, thinking: {thinking_4.content}; answer: {answer_4.content}")
        possible_answers_4.append(answer_4.content)
        thinkingmapping_4[answer_4.content] = thinking_4
        answermapping_4[answer_4.content] = answer_4
    answer_4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking_4 = thinkingmapping_4[answer_4_content]
    answer_4 = answermapping_4[answer_4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = (
        "Sub-task 5: Conduct a debate-style evaluation comparing the validated products from Sub-tasks 3 and 4 against the given multiple-choice options, "
        "critically assessing each choice to select the correct answer that matches both products A and B, ensuring no mechanistic or structural errors propagate."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_3, answer_3, thinking_4, answer_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_3, answer_3, thinking_4, answer_4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating product choice, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking5[r].append(thinking_5)
            all_answer5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct multiple-choice answer matching both products A and B.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_reflect_instruction_6 = (
        "Sub-task 6: Final reflexion step reviewing the entire reasoning chain from reaction analysis to product prediction and choice selection, "
        "confirming chemical accuracy and consistency before delivering the final answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b, thinking_1c, answer_1c, thinking_2a, answer_2a, thinking_2b, answer_2b, thinking_3, answer_3, thinking_4, answer_4, thinking_5, answer_5]
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "all previous thinking and answers"],
        "agent_collaboration": "Reflexion"
    }
    thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, reviewing entire reasoning chain, thinking: {thinking_6.content}; answer: {answer_6.content}")
    for i in range(N_max_6):
        feedback_6, correct_6 = await critic_agent_6([taskInfo, thinking_6, answer_6], "Please review the entire reasoning chain and confirm chemical accuracy and consistency, or provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback_6.content}; answer: {correct_6.content}")
        if correct_6.content == "True":
            break
        cot_inputs_6.extend([thinking_6, answer_6, feedback_6])
        thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final reflexion, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs
