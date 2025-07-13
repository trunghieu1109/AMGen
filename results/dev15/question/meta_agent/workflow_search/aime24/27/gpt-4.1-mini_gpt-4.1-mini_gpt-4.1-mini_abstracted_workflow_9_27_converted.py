async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formal Definition and Modular Conditions (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the four-digit number N as digits d1 d2 d3 d4 with 1 ≤ d1 ≤ 9 and 0 ≤ d2,d3,d4 ≤ 9, "
        "and explicitly state the condition that changing any one digit of N to 1 produces a valid four-digit number divisible by 7. "
        "Emphasize that the digit replacement applies independently to each digit, regardless of its original value, and that the resulting number must remain four-digit (no leading zeros)."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage0_subtask1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining N and conditions, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Derive modular arithmetic expressions representing the divisibility by 7 conditions for each digit replacement, "
        "explicitly formulating the constraints on N and its digits. Express these constraints in terms of d1, d2, d3, d4 and modular congruences, "
        "ensuring clarity and correctness. Treat each digit replacement condition separately."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage0_subtask2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, deriving modular conditions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Search and Verification with SC_CoT and Reflexion
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Develop a systematic, executable search algorithm that enumerates candidate four-digit numbers N in descending order (from 9999 down to 1000) "
        "and tests each candidate against the modular divisibility conditions derived in Stage 0. The search must incorporate the verification step internally to ensure only candidates satisfying all digit-replacement divisibility conditions are accepted. "
        "Implement a complete enumeration and testing procedure."
    )
    N_sc = self.max_sc
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_1_1 = []
    possible_thinkings_1_1 = []

    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, searching candidates, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent and correct candidate N for the problem.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Verification with Reflexion and retry loop
    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Verify that the candidate N found by the search algorithm indeed satisfies the divisibility conditions for all digit replacements to 1. "
        "This verification must produce a clear pass/fail output. If verification fails, the workflow must trigger a retry of the search to find the next valid candidate. "
        "Avoid proceeding with invalid candidates or ignoring verification failures. " + reflect_inst_1_2
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round

    cot_inputs_1_2 = [taskInfo, thinking_1_1, answer_1_1]

    subtask_desc_1_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Reflexion"
    }

    for attempt in range(N_max_1_2):
        thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, attempt, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, verifying candidate, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")

        feedback_1_2, correct_1_2 = await critic_agent_1_2([taskInfo, thinking_1_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", attempt, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, providing feedback, thinking: {feedback_1_2.content}; answer: {correct_1_2.content}")

        if correct_1_2.content.strip() == "True":
            sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content} (Verified)")
            subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
            logs.append(subtask_desc_1_2)
            print("Step 1.2: ", sub_tasks[-1])
            break

        cot_inputs_1_2.extend([thinking_1_2, feedback_1_2])

        # Retry search: regenerate candidate
        thinking_1_1_retry, answer_1_1_retry = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Retry search for next valid candidate N after verification failure.", is_sub_task=True)
        agents.append(f"Retry Final Decision agent, searching next candidate, thinking: {thinking_1_1_retry.content}; answer: {answer_1_1_retry.content}")
        cot_inputs_1_2 = [taskInfo, thinking_1_1_retry, answer_1_1_retry]

        if attempt == N_max_1_2 - 1:
            raise RuntimeError("Failed to find a valid candidate N after maximum retries.")

    # Stage 2: Compute Q, R and Q+R (Debate)
    debate_instr_2_1 = (
        "Sub-task 1: Express the verified number N in terms of quotient Q and remainder R when divided by 1000, i.e., compute Q = floor(N/1000) and R = N mod 1000. "
        "Clearly identify Q and R as the thousands digit and the last three digits of N, respectively."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round

    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]

    subtask_desc_2_1 = {
        "subtask_id": "stage2_subtask1",
        "instruction": debate_instr_2_1,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_2], debate_instr_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_2] + all_thinking_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instr_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing Q and R, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)

    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 1: Synthesize and finalize Q and R computation.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Compute the sum Q + R using the values obtained from the previous subtask and prepare this result for final output. "
        "Ensure the computation is straightforward_27 and error-free."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage2_subtask2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_reflect_instruction_2_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, computing Q+R, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Stage 3: Final Presentation (SC_CoT)
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Present the final answer Q + R clearly and concisely, confirming that it corresponds to the problem's requirements. "
        "Include a brief summary of the reasoning and verification steps that led to this result to ensure transparency and correctness."
    )
    N_sc_3_1 = self.max_sc
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage3_subtask1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "SC_CoT"
    }

    possible_answers_3_1 = []
    possible_thinkings_3_1 = []

    for i in range(N_sc_3_1):
        thinking_3_1, answer_3_1 = await cot_sc_agents_3_1[i]([taskInfo, thinking_2_2, answer_2_2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, presenting final answer, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
        possible_answers_3_1.append(answer_3_1)
        possible_thinkings_3_1.append(thinking_3_1)

    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_thinkings_3_1, "Sub-task 1: Synthesize and finalize the presentation of the final answer Q+R.", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Subtask 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
