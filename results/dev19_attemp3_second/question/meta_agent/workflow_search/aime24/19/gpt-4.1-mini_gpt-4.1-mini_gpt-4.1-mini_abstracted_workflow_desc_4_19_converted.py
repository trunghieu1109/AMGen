async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = "Sub-task 1: Derive and validate a polynomial representation for the expression inside the product, i.e., express 2 - 2x + x^2 as a polynomial and understand its roots and behavior over the 13th roots of unity."
    N_sc = self.max_sc
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, analyzing polynomial expression, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1 + possible_answers_0_1, "Sub-task 1: Synthesize and choose the most consistent polynomial representation and behavior.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0_subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = "Sub-task 2: Relate the product over all 13th roots of unity of the polynomial values to a known polynomial identity or resultant, such as expressing the product as the evaluation of a polynomial at a root or as a norm in a cyclotomic field, based on the output from Sub-task 1."
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, relating product to polynomial identity, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_0_2 + possible_answers_0_2, "Sub-task 2: Synthesize and choose the most consistent polynomial identity relating the product.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0_subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_1_1 = "Sub-task 1: Identify and verify the polynomial whose roots correspond to the values omega^k, and confirm the factorization or resultant expression that simplifies the product, based on the output from stage_0_subtask_2. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_2, answer_0_2], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying polynomial roots and factorization, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_1[r].append(thinking)
            all_answer_1_1[r].append(answer)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[-1] + all_answer_1_1[-1], "Sub-task 1: Final decision on polynomial roots and factorization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1_subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_2_1 = "Sub-task 1: Decompose and simplify the product expression using algebraic manipulation, cyclotomic polynomial properties, and known identities to reduce the product to a computable numeric value, based on the output from stage_1_subtask_1. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_1, answer_1_1], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying product expression, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_1[r].append(thinking)
            all_answer_2_1[r].append(answer)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_1, answer_1_1] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Final decision on simplified numeric value of product.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2_subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = "Sub-task 1: Aggregate the simplified numeric value and compute the remainder when divided by 1000, based on outputs from stage_0_subtask_1 and stage_2_subtask_1."
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_3_1[i]([taskInfo, thinking_0_1, answer_0_1, thinking_2_1, answer_2_1], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, computing remainder modulo 1000, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_3_1.append(answer)
        possible_thinkings_3_1.append(thinking)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo, thinking_0_1, answer_0_1, thinking_2_1, answer_2_1] + possible_thinkings_3_1 + possible_answers_3_1, "Sub-task 1: Final aggregation and remainder computation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_3_subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
