async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Candidate primes and modular enumeration

    # Subtask 1: Identify candidate primes p ≡ 1 (mod 8) up to a reasonable bound (e.g., 100)
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and list all candidate primes p such that p ≡ 1 (mod 8) up to 100 for testing. "
        "Do not perform any modular arithmetic checks here; only produce the candidate primes list."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, listing candidate primes p ≡ 1 (mod 8), thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    candidate_primes_text = answer_0_1.content

    # Parse candidate primes from answer (expecting a list or comma-separated primes)
    import re
    primes_found = list(map(int, re.findall(r'\b\d+\b', candidate_primes_text)))

    # Subtask 2: For each candidate prime p, exhaustively enumerate all n in [1, p-1] and find all n with n^4 ≡ -1 (mod p)
    cot_sc_instruction_0_2 = (
        "Sub-task 2: For each candidate prime p from Sub-task 1, exhaustively enumerate all integers n in the range 1 to p-1. "
        "Compute n^4 modulo p and document all n satisfying n^4 ≡ -1 (mod p). This exhaustive search is mandatory and must be complete before any conclusions about p are drawn. "
        "No early stopping or heuristics allowed. Provide a dictionary mapping each prime p to the list of all such n."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, exhaustive modular enumeration for all candidate primes, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)

    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instruction_0_2 = (
        "Sub-task 2: Given all enumerations from previous agents, synthesize and produce a consistent dictionary mapping each candidate prime p to the complete list of all n in [1, p-1] satisfying n^4 ≡ -1 (mod p). "
        "Ensure no prime is omitted and no partial results are accepted. Output in a clear JSON-like format."
    )
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, synth_instruction_0_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    agents.append(f"Final Decision agent, synthesizing exhaustive modular enumeration, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")

    # Parse the dictionary from answer_0_2.content
    import json
    try:
        modular_solutions_p = json.loads(answer_0_2.content)
    except Exception:
        modular_solutions_p = {}

    # Subtask 3a: Identify the least prime p for which there exists at least one n with n^4 ≡ -1 (mod p)
    cot_reflect_instruction_0_3a = (
        "Sub-task 3a: From the complete modular enumeration data, identify the least prime p for which there exists at least one n satisfying n^4 ≡ -1 (mod p). "
        "Do not attempt lifting to modulo p^2 here. Output the prime p and the list of all such n modulo p."
    )
    cot_agent_0_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3a = {
        "subtask_id": "stage_0.subtask_3a",
        "instruction": cot_reflect_instruction_0_3a,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_3a, answer_0_3a = await cot_agent_0_3a([taskInfo, thinking_0_2], cot_reflect_instruction_0_3a, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_3a.id}, selecting minimal prime p with solutions modulo p, thinking: {thinking_0_3a.content}; answer: {answer_0_3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_0_3a.content}; answer - {answer_0_3a.content}")
    subtask_desc_0_3a['response'] = {"thinking": thinking_0_3a, "answer": answer_0_3a}
    logs.append(subtask_desc_0_3a)

    # Parse minimal prime p and solutions n modulo p from answer_0_3a
    # Expecting output like: "The least prime p is 17 with solutions n = [6, 11]"
    import ast
    minimal_p = None
    solutions_mod_p = []
    try:
        lines = answer_0_3a.content.split('\n')
        for line in lines:
            if 'prime' in line and 'p' in line:
                nums = list(map(int, re.findall(r'\b\d+\b', line)))
                if nums:
                    minimal_p = nums[0]
            if 'solutions' in line or 'n' in line:
                list_match = re.search(r'\[(.*?)\]', line)
                if list_match:
                    solutions_mod_p = list(map(int, list_match.group(1).split(',')))
    except Exception:
        minimal_p = None
        solutions_mod_p = []

    # Subtask 3b: For the prime p identified, attempt to lift each solution n modulo p to modulo p^2
    cot_sc_instruction_0_3b = (
        f"Sub-task 3b: For the prime p={minimal_p} identified in Sub-task 3a, attempt to lift each solution n modulo p to solutions modulo p^2 by checking n^4 ≡ -1 (mod p^2). "
        "Use Hensel's lemma or direct computation as appropriate. Document all n modulo p^2 satisfying the congruence. "
        "If no solutions exist modulo p^2, indicate so explicitly."
    )
    cot_agents_0_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3b = []
    possible_thinkings_0_3b = []
    subtask_desc_0_3b = {
        "subtask_id": "stage_0.subtask_3b",
        "instruction": cot_sc_instruction_0_3b,
        "context": ["user query", thinking_0_3a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3b[i]([taskInfo, thinking_0_3a], cot_sc_instruction_0_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3b[i].id}, lifting solutions modulo p to p^2, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3b.append(answer_i)
        possible_thinkings_0_3b.append(thinking_i)

    final_decision_agent_0_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instruction_0_3b = (
        "Sub-task 3b: Given all lifting attempts from previous agents, synthesize and produce a consistent list of all n modulo p^2 satisfying n^4 ≡ -1 (mod p^2) for the prime p identified. "
        "If no solutions exist, state so explicitly."
    )
    thinking_0_3b, answer_0_3b = await final_decision_agent_0_3b([taskInfo] + possible_thinkings_0_3b, synth_instruction_0_3b, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_0_3b.content}; answer - {answer_0_3b.content}")
    subtask_desc_0_3b['response'] = {"thinking": thinking_0_3b, "answer": answer_0_3b}
    logs.append(subtask_desc_0_3b)
    agents.append(f"Final Decision agent, synthesizing lifting solutions modulo p^2, thinking: {thinking_0_3b.content}; answer: {answer_0_3b.content}")

    # Subtask 3c: Reflexion loop to confirm minimal prime p with solutions modulo p^2
    reflect_inst_0_3c = (
        "Sub-task 3c: Confirm that the prime p identified in Sub-task 3a and verified in Sub-task 3b is indeed the least prime for which there exists n with n^4 ≡ -1 (mod p^2). "
        "If no solutions modulo p^2 exist for this p, proceed to the next candidate prime and repeat Sub-tasks 3a, 3b, and 3c until the minimal prime is found. "
        "This subtask enforces an iterative critic loop to ensure no premature conclusions."
    )
    cot_agent_0_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_0_3c = self.max_round
    cot_inputs_0_3c = [taskInfo, thinking_0_3a, thinking_0_3b]
    subtask_desc_0_3c = {
        "subtask_id": "stage_0.subtask_3c",
        "instruction": reflect_inst_0_3c,
        "context": ["user query", thinking_0_3a.content, thinking_0_3b.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_3c, answer_0_3c = await cot_agent_0_3c(cot_inputs_0_3c, reflect_inst_0_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_3c.id}, confirming minimal prime p with solutions modulo p^2, thinking: {thinking_0_3c.content}; answer: {answer_0_3c.content}")
    critic_inst_0_3c = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_0_3c):
        feedback, correct = await critic_agent_0_3c([taskInfo, thinking_0_3c], critic_inst_0_3c, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_0_3c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_0_3c.extend([thinking_0_3c, feedback])
        thinking_0_3c, answer_0_3c = await cot_agent_0_3c(cot_inputs_0_3c, reflect_inst_0_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_3c.id}, refining confirmation of minimal prime p, thinking: {thinking_0_3c.content}; answer: {answer_0_3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking_0_3c.content}; answer - {answer_0_3c.content}")
    subtask_desc_0_3c['response'] = {"thinking": thinking_0_3c, "answer": answer_0_3c}
    logs.append(subtask_desc_0_3c)

    # Parse final confirmed minimal prime p and solutions modulo p^2 from answer_0_3b and answer_0_3c
    # For downstream subtasks, pass minimal_p and solutions modulo p^2

    # Stage 1: Analyze full set of positive integers n modulo p^2 satisfying n^4 ≡ -1 (mod p^2)
    cot_instruction_1_1 = (
        f"Stage 1 Sub-task 1: For the confirmed minimal prime p={minimal_p}, analyze the full set of positive integers n modulo p^2 that satisfy n^4 ≡ -1 (mod p^2). "
        "Determine the number of distinct solutions, their structure, and properties relevant to identifying the minimal such n. "
        "Use the explicit solution data passed from stage 0 to ensure consistency."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3b.content, thinking_0_3c.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3b, thinking_0_3c], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, analyzing solution set modulo p^2, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Stage 2: Find the least positive integer m such that m^4 + 1 divisible by p^2
    cot_reflect_instruction_2_1 = (
        f"Stage 2 Sub-task 1: Using the solution set characterized in Stage 1 Sub-task 1, find the least positive integer m such that m^4 + 1 is divisible by p^2, where p={minimal_p}. "
        "Explicitly use the modular solutions and verify minimality by checking all candidates up to the smallest solution found. Avoid assumptions or skipping candidates."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_1.content, thinking_0_3c.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_1, thinking_0_3c], cot_reflect_instruction_2_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, finding minimal m with m^4 + 1 divisible by p^2, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
