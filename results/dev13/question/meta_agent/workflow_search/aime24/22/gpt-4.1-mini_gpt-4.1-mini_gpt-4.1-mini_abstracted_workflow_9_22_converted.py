async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalize constraints using Chain-of-Thought
    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent the list as an ordered sequence of positive integers with unknown even length n, "
        "express the sum constraint as sum of all elements equals 30, and emphasize n is even due to median condition. "
        "Avoid assuming fixed n."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing list and sum constraint, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Formally define the unique mode condition: integer 9 appears more times than any other integer, "
        "frequency of 9 at least 2, strictly greater than others."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, defining unique mode condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Formally characterize the median condition: list length n even, median is average of two middle elements, "
        "median is positive integer not in list, median = (a_{n/2} + a_{n/2+1})/2, median integer not equal to any element."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, formalizing median condition, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Stage 1: Enumerate plausible lengths, frequencies, median pairs, construct candidates, filter strictly

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Enumerate all plausible even list lengths n (including 4, 6, 8, 10) that could satisfy sum and median conditions, "
        "reason about constraints to exclude lengths if applicable."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating plausible list lengths, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize plausible list lengths n", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_1.id}, synthesizing plausible list lengths, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Determine minimum frequency of 9 required to be unique mode for each candidate n, "
        "analyze how this constrains max frequency of others and total elements."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_0_2, answer_0_2, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, determining min frequency of 9, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize min frequency of 9 for unique mode", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_2.id}, synthesizing min frequency of 9, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Systematically enumerate all candidate pairs of middle elements (a,b) for each n such that median m=(a+b)/2 is positive integer not in list, "
        "include all valid pairs including (7,9) yielding median 8, consistent with ordering and median condition."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_0_3.content, answer_0_3.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_0_3, answer_0_3, thinking_1_1, answer_1_1], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, enumerating median pairs, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 3: Synthesize valid median pairs", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_3.id}, synthesizing median pairs, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    reflect_inst_1_4 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_4 = (
        "Sub-task 4: Construct candidate lists for each valid n and median pair (a,b) with 9 appearing at least minimum frequency, "
        "generate all possible combinations of other elements (frequency ≤1) summing with 9s to 30, ensure unique mode and median conditions, "
        "systematically explore search space without premature pruning. " + reflect_inst_1_4
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_4 = self.max_round
    cot_inputs_1_4 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2, thinking_1_3, answer_1_3]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_reflect_instruction_1_4,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content, thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_1_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, constructing candidate lists, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    critic_inst_1_4 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_1_4):
        feedback_1_4, correct_1_4 = await critic_agent_1_4([taskInfo, thinking_1_4, answer_1_4], critic_inst_1_4, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_4.id}, providing feedback, thinking: {feedback_1_4.content}; answer: {correct_1_4.content}")
        if correct_1_4.content == "True":
            break
        cot_inputs_1_4.extend([thinking_1_4, answer_1_4, feedback_1_4])
        thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_1_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, refining candidate lists, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)

    debate_instr_1_5 = (
        "Sub-task 5: Implement automated constraint satisfaction and validation filtering candidate lists from Sub-task 4, "
        "strictly enforce sum=30, unique mode=9, median integer not in list, reject any violations or near misses. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_5 = self.max_round
    all_thinking_1_5 = [[] for _ in range(N_max_1_5)]
    all_answer_1_5 = [[] for _ in range(N_max_1_5)]
    subtask_desc_1_5 = {
        "subtask_id": "stage_1.subtask_5",
        "instruction": debate_instr_1_5,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_5):
        for i, agent in enumerate(debate_agents_1_5):
            if r == 0:
                thinking_1_5, answer_1_5 = await agent([taskInfo, thinking_1_4, answer_1_4], debate_instr_1_5, r, is_sub_task=True)
            else:
                input_infos_1_5 = [taskInfo, thinking_1_4, answer_1_4] + all_thinking_1_5[r-1] + all_answer_1_5[r-1]
                thinking_1_5, answer_1_5 = await agent(input_infos_1_5, debate_instr_1_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, filtering candidates, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
            all_thinking_1_5[r].append(thinking_1_5)
            all_answer_1_5[r].append(answer_1_5)
    final_decision_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_5, answer_1_5 = await final_decision_agent_1_5([taskInfo] + all_thinking_1_5[-1] + all_answer_1_5[-1], "Sub-task 5: Final filtering of valid candidate lists. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_5.id}, final filtering of candidates, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Sub-task 1.5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking_1_5, "answer": answer_1_5}
    logs.append(subtask_desc_1_5)

    # Stage 2: Analyze and confirm valid list configurations

    debate_instr_2_1 = (
        "Sub-task 1: Analyze and simplify numeric relationships in filtered candidate lists from Sub-task 1.5, "
        "verify all constraints hold simultaneously, identify all valid configurations exactly. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_2_1,
        "context": ["user query", thinking_1_5.content, answer_1_5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_5, answer_1_5], debate_instr_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_5, answer_1_5] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instr_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing valid configurations, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Confirm valid list configurations. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_1.id}, confirming valid configurations, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Confirm and finalize the valid list configuration(s) identified in Sub-task 2.1, "
        "prepare these configurations for final computation, ensure no ambiguity about correctness and uniqueness. " + reflect_inst_2_2
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, finalizing valid configurations, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    critic_inst_2_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], critic_inst_2_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining final configurations, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    # Stage 3: Compute sum of squares of confirmed valid list(s)

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Compute the sum of squares of all items in the confirmed valid list(s) from Sub-task 2.2, "
        "provide final answer clearly and verify calculation accuracy."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_2, answer_2_2], cot_sc_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, computing sum of squares, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
