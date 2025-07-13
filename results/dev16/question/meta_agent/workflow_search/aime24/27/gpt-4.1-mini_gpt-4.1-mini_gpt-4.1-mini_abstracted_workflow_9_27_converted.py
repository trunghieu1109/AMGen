async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalize modular constraints (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent the four-digit number N as digits d1 d2 d3 d4, "
        "where d1 in [1..9], d2,d3,d4 in [0..9]. Define the four numbers obtained by changing exactly one digit of N to 1 (one at a time). "
        "Write down the four divisibility conditions that each resulting number is divisible by 7 as equations involving d1,d2,d3,d4. "
        "Avoid solving or simplifying at this stage.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage0_subtask1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing modular constraints, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Derive modular arithmetic congruences modulo 7 from the divisibility conditions established in Sub-task 1. "
        "Express each condition as a congruence involving d1,d2,d3,d4 modulo 7. State all modular equations and identify any immediate simplifications or relationships. "
        "Do not enumerate or guess digit values here.")
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage0_subtask2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, deriving modular congruences, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Analyze the modular congruences from Sub-task 2 to identify dependencies and reduce the system. "
        "Derive explicit constraints or formulas for some digits in terms of others. Prepare the system for systematic search by isolating variables or expressing digits in modular terms. "
        "Avoid numeric solutions; focus on structural simplification.")
    N_sc = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage0_subtask3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, analyzing modular constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent modular constraints simplification.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    agents.append(f"Final Decision agent {final_decision_agent_0_3.id}, synthesizing modular constraints, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Systematic search and verification
    reflexion_instruction_1_1 = (
        "Sub-task 1: Implement a systematic search for all four-digit numbers N = d1 d2 d3 d4 satisfying the modular constraints from Stage 0 Sub-task 3. "
        "Enumerate d1 from 9 down to 1, and for each d1, enumerate d2,d3,d4 in [0..9]. For each candidate, explicitly construct the four modified numbers obtained by changing each digit to 1 and verify their divisibility by 7. "
        "Collect all candidates passing all four divisibility tests. Output the maximal such candidate N along with its digits in a structured JSON format. "
        "Avoid assumptions or partial checks; all four conditions must be verified explicitly.")
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": reflexion_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3], reflexion_instruction_1_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, systematic search for maximal N, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Verify the maximal candidate N identified in Sub-task 1 by explicitly listing the four modified numbers (each with one digit replaced by 1) and computing their remainders modulo 7. "
        "Confirm that all four are divisible by 7. If any fail, reject the candidate and report failure explicitly. This verification must be detailed and explicit.")
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, verifying candidate N, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize verification results and confirm candidate validity.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    agents.append(f"Final Decision agent {final_decision_agent_1_2.id}, synthesizing verification, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    print("Step 1.2: ", sub_tasks[-1])

    # Extract candidate N and digits from verified answer
    import json
    try:
        candidate_data = json.loads(answer_1_2.content)
        N_candidate = candidate_data.get("N")
        d1 = candidate_data.get("d1")
        d2 = candidate_data.get("d2")
        d3 = candidate_data.get("d3")
        d4 = candidate_data.get("d4")
    except Exception:
        N_candidate = None
        d1 = d2 = d3 = d4 = None

    # Stage 2: Compute Q and R (Debate)
    debate_instruction_2_1 = (
        "Sub-task 1: Given the verified maximal candidate N = d1 d2 d3 d4 from Stage 1 Sub-task 2, express N in terms of quotient Q and remainder R when divided by 1000. "
        "Explicitly identify Q = d1 and R = 100*d2 + 10*d3 + d4. Provide these values clearly and avoid ambiguity.")
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage2_subtask1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2] + all_thinking_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing Q and R, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 1: Synthesize and finalize Q and R values.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Stage 2 Sub-task 2: Compute Q + R (Reflexion)
    reflexion_instruction_2_2 = (
        "Sub-task 2: Compute the sum Q + R using the values obtained in Sub-task 1. "
        "Present the calculation step-by-step and prepare the final numeric result for output. Avoid new variables or assumptions.")
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage2_subtask2",
        "instruction": reflexion_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1], reflexion_instruction_2_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, computing Q+R, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Stage 3: Final consistency check (SC_CoT)
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Perform a final consistency check on the computed value Q + R to ensure it aligns with the problem statement and the verified candidate N. "
        "Confirm all previous steps are consistent and the final answer is justified. Document verification explicitly. Avoid silent assumptions.")
    N_sc_3_1 = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_1)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage3_subtask1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3_1):
        thinking_i, answer_i = await cot_agents_3_1[i]([taskInfo, thinking_2_2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, final consistency check, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_thinkings_3_1, "Sub-task 1: Synthesize final consistency check and confirm final answer.", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    agents.append(f"Final Decision agent {final_decision_agent_3_1.id}, final consistency check, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
