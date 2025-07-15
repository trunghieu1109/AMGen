async def forward_178(self, taskInfo):
    from collections import Counter
    import numpy as np
    import scipy.linalg

    sub_tasks = []
    agents = []
    logs = []

    matrices_str = {
        'W': '0, 0, 1; 0, 1, 0; 1, 0, 0',
        'X': '1j, -1, 2j; 1, 0, 1; 2j, -1, -1j',
        'Y': '0.5, 0.1, 0.2; 0.1, 0.25, 0.1; 0.2, 0.1, 0.25',
        'Z': '3, 2j, 5; -2j, -2, -4j; 5, 4j, 4'
    }

    def parse_matrix(s):
        rows = s.split(';')
        matrix = []
        for r in rows:
            entries = r.strip().split(',')
            row = []
            for e in entries:
                e = e.strip()
                if 'j' in e or 'i' in e:
                    e = e.replace('i', 'j')
                    val = complex(e)
                else:
                    val = float(e)
                row.append(val)
            matrix.append(row)
        return np.array(matrix, dtype=complex)

    W = parse_matrix(matrices_str['W'])
    X = parse_matrix(matrices_str['X'])
    Y = parse_matrix(matrices_str['Y'])
    Z = parse_matrix(matrices_str['Z'])

    # Stage 1: Verify Hermiticity and anti-Hermiticity of W, X, Y, Z using Debate
    debate_instr_1 = (
        "Sub-task 1: Verify Hermiticity and anti-Hermiticity properties of matrices W, X, Y, and Z by performing element-wise conjugate transpose comparisons. "
        "State clearly which matrices are Hermitian, anti-Hermitian, or neither. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round

    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]

    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1] + all_answer_1[-1],
                                                    "Sub-task 1: Synthesize and choose the most consistent and correct solutions for Hermiticity and anti-Hermiticity verification." +
                                                    "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                    is_sub_task=True)

    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Parse subtask 1 answer to extract Hermiticity results
    # We expect answer1.content to contain which matrices are Hermitian, anti-Hermitian, or neither
    # For safety, we parse it manually here

    hermitian = {}
    anti_hermitian = {}
    neither = {}

    # Simple heuristic parsing
    content_lower = answer1.content.lower()
    for mat in ['w', 'x', 'y', 'z']:
        if f'{mat} is hermitian' in content_lower:
            hermitian[mat.upper()] = True
            anti_hermitian[mat.upper()] = False
            neither[mat.upper()] = False
        elif f'{mat} is anti-hermitian' in content_lower or f'{mat} is anti hermitian' in content_lower:
            hermitian[mat.upper()] = False
            anti_hermitian[mat.upper()] = True
            neither[mat.upper()] = False
        elif f'{mat} is neither' in content_lower or f'{mat} is not hermitian' in content_lower:
            hermitian[mat.upper()] = False
            anti_hermitian[mat.upper()] = False
            neither[mat.upper()] = True
        else:
            # fallback: compute directly
            M = {'W': W, 'X': X, 'Y': Y, 'Z': Z}[mat.upper()]
            M_dag = M.conj().T
            if np.allclose(M, M_dag):
                hermitian[mat.upper()] = True
                anti_hermitian[mat.upper()] = False
                neither[mat.upper()] = False
            elif np.allclose(M, -M_dag):
                hermitian[mat.upper()] = False
                anti_hermitian[mat.upper()] = True
                neither[mat.upper()] = False
            else:
                hermitian[mat.upper()] = False
                anti_hermitian[mat.upper()] = False
                neither[mat.upper()] = True

    # Stage 2: Verify unitarity of W and e^X, positivity and unit trace of Y, Hermiticity of Z using SC_CoT
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the Hermiticity and anti-Hermiticity results from Sub-task 1, verify the unitarity of matrices W and e^X (matrix exponential of X). "
        "Check positivity and unit trace of Y to confirm if it can represent a quantum state (density matrix). "
        "Also verify if Z is Hermitian and thus a valid observable candidate. "
        "Incorporate the anti-Hermiticity result of X to correctly assess unitarity of e^X. "
        "Perform eigenvalue computations for positivity checks."
    )

    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]

    possible_answers_2 = []
    possible_thinkings_2 = []

    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2,
                                                    "Sub-task 2: Synthesize and choose the most consistent and correct solutions for unitarity, positivity, and Hermiticity verification.",
                                                    is_sub_task=True)

    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Compute similarity transform (e^X)*Y*(e^{-X}) and verify if it represents a valid quantum state using SC_CoT
    cot_sc_instruction_3 = (
        "Sub-task 3: Compute the similarity transform (e^X)*Y*(e^{-X}) and verify whether the resulting matrix represents a valid quantum state. "
        "Check Hermiticity, positivity (via eigenvalues), and unit trace. "
        "Incorporate the unitarity result of e^X from Sub-task 2 to determine if positivity is preserved under the similarity transform. "
        "If e^X is unitary, positivity is preserved; otherwise, explicit positivity verification is mandatory."
    )

    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]

    possible_answers_3 = []
    possible_thinkings_3 = []

    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings_3 + possible_answers_3,
                                                    "Sub-task 3: Synthesize and choose the most consistent and correct solutions for similarity transform and quantum state verification.",
                                                    is_sub_task=True)

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Evaluate correctness of each given statement (choices 1 to 4) using Debate
    debate_instr_4 = (
        "Sub-task 4: Evaluate the correctness of each given statement (choices 1 to 4) based on the verified properties and computations from all previous subtasks. "
        "Reconcile any contradictions, especially regarding unitarity of e^X and positivity of the similarity-transformed matrix. "
        "Explicitly reference prior results and ensure logical coherence. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round

    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]

    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_4,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + all_thinking_4[-1] + all_answer_4[-1],
                                                    "Sub-task 4: Synthesize and choose the most consistent and correct evaluation of the given statements." +
                                                    "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                    is_sub_task=True)

    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
